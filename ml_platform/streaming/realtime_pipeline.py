"""
Real-time Data Pipeline for NBA Analytics
Kafka/Kinesis integration with stream processing using Flink/Spark Streaming
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from confluent_kafka import Producer, Consumer, KafkaError, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import boto3
from prometheus_client import Counter, Histogram, Gauge
import logging
import redis
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import RealDictCursor

# Metrics
events_processed = Counter('streaming_events_processed_total', 'Total events processed', ['event_type'])
processing_latency = Histogram('streaming_processing_latency_seconds', 'Processing latency')
lag_gauge = Gauge('streaming_consumer_lag', 'Consumer lag', ['topic', 'partition'])
error_counter = Counter('streaming_errors_total', 'Processing errors', ['error_type'])

logger = logging.getLogger(__name__)


@dataclass
class GameEvent:
    """Real-time game event"""
    event_id: str
    game_id: str
    timestamp: datetime
    event_type: str  # 'shot', 'rebound', 'assist', 'turnover', etc.
    player_id: str
    team_id: str
    details: Dict[str, Any]
    game_clock: str
    quarter: int
    score_home: int
    score_away: int


@dataclass
class PlayerState:
    """Real-time player state"""
    player_id: str
    game_id: str
    minutes_played: float
    points: int
    rebounds: int
    assists: int
    field_goals_made: int
    field_goals_attempted: int
    last_updated: datetime
    momentum_score: float
    fatigue_index: float


class RealtimeDataPipeline:
    """Production real-time data pipeline with Kafka and stream processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Kafka configuration
        self.producer_config = {
            'bootstrap.servers': config['kafka_brokers'],
            'compression.type': 'snappy',
            'linger.ms': 5,
            'batch.size': 32768,
            'buffer.memory': 67108864,
            'retries': 3,
            'enable.idempotence': True
        }
        
        self.consumer_config = {
            'bootstrap.servers': config['kafka_brokers'],
            'group.id': config['consumer_group'],
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False,
            'max.poll.interval.ms': 300000,
            'session.timeout.ms': 45000
        }
        
        # Initialize Kafka clients
        self.producer = Producer(self.producer_config)
        self.admin_client = AdminClient({'bootstrap.servers': config['kafka_brokers']})
        
        # Schema Registry for Avro
        self.schema_registry = SchemaRegistryClient({
            'url': config['schema_registry_url']
        })
        
        # State store (Redis)
        self.redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # PostgreSQL for event sourcing
        self.pg_conn = psycopg2.connect(config['postgres_url'])
        
        # Kinesis client (for AWS deployments)
        self.kinesis_client = boto3.client('kinesis', region_name=config['aws_region'])
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Initialize topics
        self._create_topics()
    
    def _create_topics(self):
        """Create Kafka topics with proper configuration"""
        topics = [
            NewTopic('game-events', num_partitions=12, replication_factor=3),
            NewTopic('player-stats', num_partitions=12, replication_factor=3),
            NewTopic('predictions', num_partitions=6, replication_factor=3),
            NewTopic('alerts', num_partitions=3, replication_factor=3),
            NewTopic('feature-updates', num_partitions=6, replication_factor=3)
        ]
        
        # Create topics
        fs = self.admin_client.create_topics(topics)
        
        for topic, f in fs.items():
            try:
                f.result()
                logger.info(f"Topic {topic} created")
            except Exception as e:
                logger.warning(f"Topic {topic} creation failed: {e}")
    
    async def ingest_game_event(self, event: GameEvent) -> None:
        """Ingest real-time game event"""
        try:
            # Serialize event
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            
            # Send to Kafka
            self.producer.produce(
                topic='game-events',
                key=event.player_id,
                value=json.dumps(event_data),
                callback=self._delivery_callback
            )
            
            # Also send to Kinesis for AWS integrations
            if self.config.get('use_kinesis'):
                self.kinesis_client.put_record(
                    StreamName='nba-game-events',
                    Data=json.dumps(event_data),
                    PartitionKey=event.player_id
                )
            
            # Update player state
            await self._update_player_state(event)
            
            # Trigger feature computation
            await self._trigger_feature_update(event)
            
            events_processed.labels(event_type=event.event_type).inc()
            
        except Exception as e:
            error_counter.labels(error_type='ingestion_error').inc()
            logger.error(f"Failed to ingest event: {e}")
    
    async def _update_player_state(self, event: GameEvent) -> None:
        """Update real-time player state"""
        state_key = f"player_state:{event.player_id}:{event.game_id}"
        
        # Get current state
        current_state_json = self.redis_client.get(state_key)
        
        if current_state_json:
            current_state = PlayerState(**json.loads(current_state_json))
        else:
            current_state = PlayerState(
                player_id=event.player_id,
                game_id=event.game_id,
                minutes_played=0,
                points=0,
                rebounds=0,
                assists=0,
                field_goals_made=0,
                field_goals_attempted=0,
                last_updated=event.timestamp,
                momentum_score=0.0,
                fatigue_index=0.0
            )
        
        # Update based on event type
        if event.event_type == 'shot':
            if event.details.get('made'):
                current_state.points += event.details.get('points', 2)
                current_state.field_goals_made += 1
            current_state.field_goals_attempted += 1
            
        elif event.event_type == 'rebound':
            current_state.rebounds += 1
            
        elif event.event_type == 'assist':
            current_state.assists += 1
        
        # Calculate derived metrics
        current_state.momentum_score = self._calculate_momentum(current_state, event)
        current_state.fatigue_index = self._calculate_fatigue(current_state, event)
        current_state.last_updated = event.timestamp
        
        # Save state
        state_data = asdict(current_state)
        state_data['last_updated'] = current_state.last_updated.isoformat()
        
        self.redis_client.setex(
            state_key,
            86400,  # 24 hour TTL
            json.dumps(state_data)
        )
        
        # Emit state update
        self.producer.produce(
            topic='player-stats',
            key=event.player_id,
            value=json.dumps(state_data)
        )
    
    def create_spark_streaming_pipeline(self) -> None:
        """Create Spark Streaming pipeline for complex analytics"""
        spark = SparkSession.builder \
            .appName("NBA Real-time Analytics") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        # Read from Kafka
        game_events_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config['kafka_brokers']) \
            .option("subscribe", "game-events") \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse events
        event_schema = StructType([
            StructField("event_id", StringType()),
            StructField("game_id", StringType()),
            StructField("timestamp", TimestampType()),
            StructField("event_type", StringType()),
            StructField("player_id", StringType()),
            StructField("team_id", StringType()),
            StructField("details", MapType(StringType(), StringType())),
            StructField("game_clock", StringType()),
            StructField("quarter", IntegerType()),
            StructField("score_home", IntegerType()),
            StructField("score_away", IntegerType())
        ])
        
        parsed_events = game_events_df.select(
            F.from_json(F.col("value").cast("string"), event_schema).alias("event")
        ).select("event.*")
        
        # Calculate real-time analytics
        
        # 1. Player scoring rate (points per minute)
        player_scoring_rate = parsed_events \
            .filter(F.col("event_type") == "shot") \
            .filter(F.col("details.made") == "true") \
            .groupBy(
                F.window(F.col("timestamp"), "5 minutes"),
                "player_id"
            ) \
            .agg(
                F.sum("details.points").alias("points_scored"),
                F.count("*").alias("shots_made")
            ) \
            .select(
                F.col("player_id"),
                F.col("window.start").alias("window_start"),
                (F.col("points_scored") / 5).alias("points_per_minute")
            )
        
        # 2. Team momentum (rolling scoring differential)
        team_momentum = parsed_events \
            .groupBy(
                F.window(F.col("timestamp"), "2 minutes", "30 seconds"),
                "team_id"
            ) \
            .agg(
                F.sum(F.when(F.col("event_type") == "shot", F.col("details.points")).otherwise(0)).alias("points"),
                F.count(F.when(F.col("event_type") == "turnover", 1)).alias("turnovers")
            ) \
            .select(
                F.col("team_id"),
                F.col("window.start").alias("window_start"),
                (F.col("points") - F.col("turnovers") * 2).alias("momentum_score")
            )
        
        # 3. Hot hand detection
        hot_hand_detection = parsed_events \
            .filter(F.col("event_type") == "shot") \
            .withColumn("shot_made", F.when(F.col("details.made") == "true", 1).otherwise(0)) \
            .withColumn(
                "recent_shots",
                F.collect_list("shot_made").over(
                    Window.partitionBy("player_id").orderBy("timestamp").rowsBetween(-4, 0)
                )
            ) \
            .withColumn(
                "hot_hand",
                F.when(F.size(F.filter(F.col("recent_shots"), lambda x: x == 1)) >= 3, True).otherwise(False)
            )
        
        # Write results back to Kafka
        query1 = player_scoring_rate.writeStream \
            .outputMode("append") \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config['kafka_brokers']) \
            .option("topic", "player-analytics") \
            .option("checkpointLocation", "/tmp/player_scoring_checkpoint") \
            .trigger(processingTime='10 seconds') \
            .start()
        
        query2 = team_momentum.writeStream \
            .outputMode("append") \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config['kafka_brokers']) \
            .option("topic", "team-analytics") \
            .option("checkpointLocation", "/tmp/team_momentum_checkpoint") \
            .trigger(processingTime='10 seconds') \
            .start()
        
        # Also write to Delta Lake for historical analysis
        historical_query = parsed_events.writeStream \
            .outputMode("append") \
            .format("delta") \
            .option("checkpointLocation", "/tmp/events_checkpoint") \
            .option("path", "s3://nba-analytics/delta/game_events") \
            .partitionBy("game_id", "quarter") \
            .trigger(processingTime='1 minute') \
            .start()
        
        # Wait for termination
        spark.streams.awaitAnyTermination()
    
    def create_flink_pipeline(self) -> None:
        """Create Apache Flink pipeline for ultra-low latency processing"""
        from pyflink.datastream import StreamExecutionEnvironment
        from pyflink.table import StreamTableEnvironment, DataTypes
        from pyflink.table.descriptors import Schema, Kafka, Json
        
        # Set up environment
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(4)
        
        t_env = StreamTableEnvironment.create(env)
        
        # Define source table (Kafka)
        t_env.execute_sql("""
            CREATE TABLE game_events (
                event_id STRING,
                game_id STRING,
                event_time TIMESTAMP(3),
                event_type STRING,
                player_id STRING,
                team_id STRING,
                details MAP<STRING, STRING>,
                game_clock STRING,
                quarter INT,
                score_home INT,
                score_away INT,
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'game-events',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'flink-consumer',
                'format' = 'json',
                'scan.startup.mode' = 'latest-offset'
            )
        """)
        
        # Complex event processing - detect scoring runs
        t_env.execute_sql("""
            CREATE VIEW scoring_runs AS
            SELECT 
                team_id,
                TUMBLE_START(event_time, INTERVAL '2' MINUTE) as window_start,
                TUMBLE_END(event_time, INTERVAL '2' MINUTE) as window_end,
                COUNT(CASE WHEN event_type = 'shot' AND details['made'] = 'true' THEN 1 END) as shots_made,
                SUM(CASE WHEN event_type = 'shot' AND details['made'] = 'true' 
                    THEN CAST(details['points'] AS INT) ELSE 0 END) as points_scored,
                COUNT(CASE WHEN event_type = 'turnover' THEN 1 END) as turnovers
            FROM game_events
            GROUP BY team_id, TUMBLE(event_time, INTERVAL '2' MINUTE)
            HAVING points_scored >= 10 AND turnovers = 0
        """)
        
        # Pattern detection - find comeback situations
        t_env.execute_sql("""
            CREATE VIEW comeback_detection AS
            SELECT *
            FROM game_events
            MATCH_RECOGNIZE (
                PARTITION BY game_id
                ORDER BY event_time
                MEASURES
                    FIRST(A.team_id) as trailing_team,
                    LAST(D.score_home) - LAST(D.score_away) as final_diff,
                    FIRST(A.score_home) - FIRST(A.score_away) as initial_diff
                ONE ROW PER MATCH
                PATTERN (A B+ C+ D)
                DEFINE
                    A AS A.score_home - A.score_away < -10,
                    B AS B.event_type = 'shot' AND B.details['made'] = 'true',
                    C AS C.score_home - C.score_away > PREV(C.score_home - C.score_away),
                    D AS D.score_home - D.score_away > 0
            )
        """)
        
        # Output to alerts topic
        t_env.execute_sql("""
            CREATE TABLE alerts (
                alert_type STRING,
                team_id STRING,
                details STRING,
                timestamp TIMESTAMP(3)
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'alerts',
                'properties.bootstrap.servers' = 'localhost:9092',
                'format' = 'json'
            )
        """)
        
        # Send alerts for scoring runs
        t_env.execute_sql("""
            INSERT INTO alerts
            SELECT 
                'scoring_run' as alert_type,
                team_id,
                CONCAT('Team on ', CAST(points_scored AS STRING), '-0 run') as details,
                window_end as timestamp
            FROM scoring_runs
        """)
        
        # Execute pipeline
        env.execute("NBA Real-time Event Processing")
    
    async def implement_cdc(self) -> None:
        """Implement Change Data Capture for database sync"""
        # Using Debezium for PostgreSQL CDC
        debezium_config = {
            "name": "nba-postgres-connector",
            "config": {
                "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
                "database.hostname": "localhost",
                "database.port": "5432",
                "database.user": "postgres",
                "database.password": "password",
                "database.dbname": "nba_analytics",
                "database.server.name": "nba_db",
                "table.include.list": "public.player_stats,public.game_results",
                "plugin.name": "pgoutput",
                "publication.autocreate.mode": "filtered",
                "key.converter": "org.apache.kafka.connect.json.JsonConverter",
                "value.converter": "org.apache.kafka.connect.json.JsonConverter",
                "transforms": "route",
                "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
                "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
                "transforms.route.replacement": "cdc-$3"
            }
        }
        
        # Register connector with Kafka Connect
        import requests
        response = requests.post(
            f"{self.config['kafka_connect_url']}/connectors",
            json=debezium_config
        )
        
        if response.status_code == 201:
            logger.info("CDC connector created successfully")
        else:
            logger.error(f"Failed to create CDC connector: {response.text}")
    
    async def process_incremental_updates(self) -> None:
        """Process incremental model updates based on new data"""
        consumer = Consumer(self.consumer_config)
        consumer.subscribe(['player-stats', 'feature-updates'])
        
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Consumer error: {msg.error()}")
                    break
            
            try:
                # Parse message
                data = json.loads(msg.value().decode('utf-8'))
                
                # Check if model update is needed
                if await self._should_update_model(data):
                    # Trigger incremental learning
                    await self._trigger_incremental_learning(data)
                
                # Commit offset
                consumer.commit(asynchronous=False)
                
            except Exception as e:
                error_counter.labels(error_type='processing_error').inc()
                logger.error(f"Error processing message: {e}")
    
    async def _should_update_model(self, data: Dict[str, Any]) -> bool:
        """Determine if model should be updated based on new data"""
        # Check data drift
        current_stats = self._get_current_feature_stats()
        new_stats = self._calculate_feature_stats(data)
        
        drift_score = self._calculate_drift(current_stats, new_stats)
        
        # Update if significant drift detected
        return drift_score > 0.1
    
    async def _trigger_incremental_learning(self, data: Dict[str, Any]) -> None:
        """Trigger incremental model update"""
        update_request = {
            'model_name': 'nba_predictor',
            'update_type': 'incremental',
            'new_data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to model update service
        self.producer.produce(
            topic='model-updates',
            key='incremental',
            value=json.dumps(update_request)
        )
    
    def _calculate_momentum(self, state: PlayerState, event: GameEvent) -> float:
        """Calculate player momentum score"""
        # Simple momentum based on recent performance
        if state.field_goals_attempted > 0:
            shooting_pct = state.field_goals_made / state.field_goals_attempted
        else:
            shooting_pct = 0
        
        # Points per minute
        if state.minutes_played > 0:
            ppm = state.points / state.minutes_played
        else:
            ppm = 0
        
        momentum = (shooting_pct * 0.5 + min(ppm / 1.0, 1.0) * 0.5)
        
        return momentum
    
    def _calculate_fatigue(self, state: PlayerState, event: GameEvent) -> float:
        """Calculate player fatigue index"""
        # Simple fatigue based on minutes played
        if event.quarter <= 2:
            expected_minutes = event.quarter * 12 * 0.75  # 75% playing time
        else:
            expected_minutes = 24 * 0.75 + (event.quarter - 2) * 12 * 0.85
        
        if expected_minutes > 0:
            fatigue = min(state.minutes_played / expected_minutes, 1.5)
        else:
            fatigue = 0
        
        return fatigue
    
    def _delivery_callback(self, err, msg):
        """Kafka delivery callback"""
        if err:
            error_counter.labels(error_type='delivery_error').inc()
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def monitor_consumer_lag(self) -> None:
        """Monitor Kafka consumer lag"""
        consumer = Consumer(self.consumer_config)
        
        while True:
            # Get current position
            partitions = consumer.assignment()
            
            for partition in partitions:
                # Get committed offset
                committed = consumer.committed([partition])[0].offset
                
                # Get high water mark
                low, high = consumer.get_watermark_offsets(partition)
                
                # Calculate lag
                lag = high - committed
                
                lag_gauge.labels(
                    topic=partition.topic,
                    partition=partition.partition
                ).set(lag)
            
            time.sleep(30)  # Check every 30 seconds


# Event sourcing for game state reconstruction
class EventSourcingStore:
    """Event sourcing for complete game state reconstruction"""
    
    def __init__(self, pg_conn):
        self.conn = pg_conn
        self._create_tables()
    
    def _create_tables(self):
        """Create event sourcing tables"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS game_events (
                    event_id SERIAL PRIMARY KEY,
                    game_id VARCHAR(50),
                    event_timestamp TIMESTAMP,
                    event_type VARCHAR(50),
                    event_data JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_game_events_game_id 
                ON game_events(game_id);
                
                CREATE INDEX IF NOT EXISTS idx_game_events_timestamp 
                ON game_events(event_timestamp);
                
                CREATE TABLE IF NOT EXISTS game_snapshots (
                    snapshot_id SERIAL PRIMARY KEY,
                    game_id VARCHAR(50),
                    snapshot_timestamp TIMESTAMP,
                    game_state JSONB,
                    event_id_marker INTEGER REFERENCES game_events(event_id),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            self.conn.commit()
    
    def append_event(self, event: GameEvent) -> int:
        """Append event to event store"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO game_events (game_id, event_timestamp, event_type, event_data)
                VALUES (%s, %s, %s, %s)
                RETURNING event_id
            """, (event.game_id, event.timestamp, event.event_type, json.dumps(asdict(event))))
            
            event_id = cur.fetchone()[0]
            self.conn.commit()
            
            return event_id
    
    def reconstruct_game_state(self, game_id: str, 
                              timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Reconstruct game state at specific point in time"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find nearest snapshot
            if timestamp:
                cur.execute("""
                    SELECT * FROM game_snapshots
                    WHERE game_id = %s AND snapshot_timestamp <= %s
                    ORDER BY snapshot_timestamp DESC
                    LIMIT 1
                """, (game_id, timestamp))
            else:
                cur.execute("""
                    SELECT * FROM game_snapshots
                    WHERE game_id = %s
                    ORDER BY snapshot_timestamp DESC
                    LIMIT 1
                """, (game_id,))
            
            snapshot = cur.fetchone()
            
            if snapshot:
                game_state = snapshot['game_state']
                start_event_id = snapshot['event_id_marker']
            else:
                game_state = self._initial_game_state()
                start_event_id = 0
            
            # Apply events since snapshot
            if timestamp:
                cur.execute("""
                    SELECT * FROM game_events
                    WHERE game_id = %s AND event_id > %s AND event_timestamp <= %s
                    ORDER BY event_id
                """, (game_id, start_event_id, timestamp))
            else:
                cur.execute("""
                    SELECT * FROM game_events
                    WHERE game_id = %s AND event_id > %s
                    ORDER BY event_id
                """, (game_id, start_event_id))
            
            events = cur.fetchall()
            
            # Apply each event to reconstruct state
            for event in events:
                game_state = self._apply_event(game_state, event['event_data'])
            
            return game_state
    
    def _apply_event(self, state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply event to game state"""
        # Update state based on event type
        if event['event_type'] == 'shot':
            player_id = event['player_id']
            if player_id not in state['players']:
                state['players'][player_id] = self._initial_player_state()
            
            if event['details'].get('made'):
                state['players'][player_id]['points'] += event['details'].get('points', 2)
                state['players'][player_id]['field_goals_made'] += 1
            
            state['players'][player_id]['field_goals_attempted'] += 1
            
        # Add other event type handlers...
        
        return state
    
    def _initial_game_state(self) -> Dict[str, Any]:
        """Initial game state"""
        return {
            'game_id': None,
            'score_home': 0,
            'score_away': 0,
            'quarter': 1,
            'game_clock': '12:00',
            'players': {}
        }
    
    def _initial_player_state(self) -> Dict[str, Any]:
        """Initial player state"""
        return {
            'points': 0,
            'rebounds': 0,
            'assists': 0,
            'field_goals_made': 0,
            'field_goals_attempted': 0,
            'minutes_played': 0
        }


if __name__ == "__main__":
    config = {
        'kafka_brokers': 'localhost:9092',
        'consumer_group': 'nba-analytics',
        'schema_registry_url': 'http://localhost:8081',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'postgres_url': 'postgresql://user:pass@localhost/nba',
        'aws_region': 'us-east-1',
        'use_kinesis': False,
        'kafka_connect_url': 'http://localhost:8083'
    }
    
    pipeline = RealtimeDataPipeline(config)
    
    # Start Spark Streaming pipeline
    pipeline.create_spark_streaming_pipeline()