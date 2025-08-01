"""
Distributed Feature Computation with Apache Spark
Handles large-scale feature engineering for NBA analytics
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
from delta import DeltaTable, configure_spark_with_delta_pip

logger = logging.getLogger(__name__)


class SparkFeatureComputer:
    """Distributed feature computation engine using Spark"""
    
    def __init__(self, spark_config: Dict[str, Any]):
        # Initialize Spark with Delta Lake support
        builder = SparkSession.builder \
            .appName("NBA Feature Computation") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200")
        
        # Add custom configs
        for key, value in spark_config.items():
            builder = builder.config(key, value)
        
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        
    def compute_player_features(self, games_df: DataFrame) -> DataFrame:
        """Compute comprehensive player features"""
        
        # Basic game stats aggregations
        player_window = Window.partitionBy("player_id").orderBy("game_date")
        player_window_unbounded = Window.partitionBy("player_id").orderBy("game_date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        
        # Recent performance windows
        last_5_games = Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-4, 0)
        last_10_games = Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-9, 0)
        last_30_days = Window.partitionBy("player_id").orderBy("game_date").rangeBetween(-30 * 86400, 0)
        
        features_df = games_df \
            .withColumn("days_since_last_game", 
                F.datediff(F.col("game_date"), F.lag("game_date", 1).over(player_window))) \
            .withColumn("is_back_to_back", 
                F.when(F.col("days_since_last_game") == 1, 1).otherwise(0)) \
            .withColumn("rest_days", 
                F.when(F.col("days_since_last_game").isNull(), 3)
                 .otherwise(F.col("days_since_last_game") - 1)) \
            .withColumn("games_played_season", 
                F.row_number().over(player_window_unbounded)) \
            .withColumn("cumulative_minutes", 
                F.sum("minutes_played").over(player_window_unbounded)) \
            .withColumn("avg_points_last_5", 
                F.avg("points").over(last_5_games)) \
            .withColumn("avg_points_last_10", 
                F.avg("points").over(last_10_games)) \
            .withColumn("std_points_last_10", 
                F.stddev("points").over(last_10_games)) \
            .withColumn("max_points_last_10", 
                F.max("points").over(last_10_games)) \
            .withColumn("min_points_last_10", 
                F.min("points").over(last_10_games)) \
            .withColumn("avg_rebounds_last_5", 
                F.avg("rebounds").over(last_5_games)) \
            .withColumn("avg_assists_last_5", 
                F.avg("assists").over(last_5_games)) \
            .withColumn("avg_minutes_last_5", 
                F.avg("minutes_played").over(last_5_games)) \
            .withColumn("total_points_last_30days", 
                F.sum("points").over(last_30_days)) \
            .withColumn("games_last_30days", 
                F.count("game_id").over(last_30_days)) \
            .withColumn("shooting_pct_last_5", 
                F.sum("field_goals_made").over(last_5_games) / 
                F.sum("field_goals_attempted").over(last_5_games)) \
            .withColumn("three_point_rate_last_5", 
                F.sum("three_pointers_attempted").over(last_5_games) / 
                F.sum("field_goals_attempted").over(last_5_games)) \
            .withColumn("free_throw_rate_last_5", 
                F.sum("free_throws_attempted").over(last_5_games) / 
                F.sum("field_goals_attempted").over(last_5_games)) \
            .withColumn("usage_rate_last_5", 
                (F.sum("field_goals_attempted").over(last_5_games) + 
                 0.44 * F.sum("free_throws_attempted").over(last_5_games) + 
                 F.sum("turnovers").over(last_5_games)) / 
                F.sum("minutes_played").over(last_5_games) * 48) \
            .withColumn("efficiency_rating_last_5",
                (F.sum("points").over(last_5_games) + 
                 F.sum("rebounds").over(last_5_games) + 
                 F.sum("assists").over(last_5_games) + 
                 F.sum("steals").over(last_5_games) + 
                 F.sum("blocks").over(last_5_games) - 
                 F.sum("turnovers").over(last_5_games) - 
                 (F.sum("field_goals_attempted").over(last_5_games) - 
                  F.sum("field_goals_made").over(last_5_games)) - 
                 (F.sum("free_throws_attempted").over(last_5_games) - 
                  F.sum("free_throws_made").over(last_5_games))) / 
                F.count("game_id").over(last_5_games))
        
        # Add momentum features
        features_df = self._add_momentum_features(features_df)
        
        # Add opponent-adjusted features
        features_df = self._add_opponent_adjusted_features(features_df)
        
        # Add injury risk features
        features_df = self._add_injury_risk_features(features_df)
        
        return features_df
    
    def _add_momentum_features(self, df: DataFrame) -> DataFrame:
        """Calculate momentum and trend features"""
        player_window = Window.partitionBy("player_id").orderBy("game_date")
        
        # Calculate moving averages
        ma_3 = Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-2, 0)
        ma_10 = Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-9, 0)
        
        return df \
            .withColumn("points_ma3", F.avg("points").over(ma_3)) \
            .withColumn("points_ma10", F.avg("points").over(ma_10)) \
            .withColumn("points_momentum", 
                (F.col("points_ma3") - F.col("points_ma10")) / F.col("points_ma10")) \
            .withColumn("hot_streak", 
                F.when(F.col("points_ma3") > F.col("points_ma10") * 1.2, 1).otherwise(0)) \
            .withColumn("cold_streak", 
                F.when(F.col("points_ma3") < F.col("points_ma10") * 0.8, 1).otherwise(0)) \
            .withColumn("consistency_score",
                1 - (F.stddev("points").over(ma_10) / F.avg("points").over(ma_10)))
    
    def _add_opponent_adjusted_features(self, df: DataFrame) -> DataFrame:
        """Add opponent strength adjusted features"""
        # Calculate opponent defensive ratings
        opponent_window = Window.partitionBy("opponent_team_id")
        
        opponent_stats = df.groupBy("opponent_team_id") \
            .agg(
                F.avg("points").alias("avg_points_allowed"),
                F.avg("rebounds").alias("avg_rebounds_allowed"),
                F.avg("assists").alias("avg_assists_allowed")
            )
        
        # Join back to main dataframe
        df_with_opp = df.join(opponent_stats, on="opponent_team_id", how="left")
        
        # Calculate adjusted metrics
        return df_with_opp \
            .withColumn("points_vs_opponent_avg", 
                F.col("points") - F.col("avg_points_allowed")) \
            .withColumn("opponent_defensive_rating", 
                F.col("avg_points_allowed") / F.avg("avg_points_allowed").over(Window.partitionBy())) \
            .withColumn("expected_points_vs_opponent", 
                F.col("avg_points_last_5") * F.col("opponent_defensive_rating"))
    
    def _add_injury_risk_features(self, df: DataFrame) -> DataFrame:
        """Calculate injury risk indicators"""
        player_window = Window.partitionBy("player_id").orderBy("game_date")
        last_10_games = Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-9, 0)
        
        return df \
            .withColumn("workload_last_10", 
                F.sum("minutes_played").over(last_10_games)) \
            .withColumn("workload_spike", 
                F.col("minutes_played") / F.avg("minutes_played").over(last_10_games)) \
            .withColumn("consecutive_games", 
                F.sum(F.when(F.col("days_since_last_game") <= 2, 1).otherwise(0))
                 .over(player_window.rowsBetween(-10, 0))) \
            .withColumn("fatigue_index", 
                (F.col("workload_last_10") / 10 - 25) / 10 + 
                F.col("consecutive_games") / 10) \
            .withColumn("injury_risk_score", 
                F.when(F.col("fatigue_index") > 1.5, "high")
                 .when(F.col("fatigue_index") > 1.0, "medium")
                 .otherwise("low"))
    
    def compute_team_features(self, games_df: DataFrame) -> DataFrame:
        """Compute team-level features"""
        team_window = Window.partitionBy("team_id").orderBy("game_date")
        last_5_games = Window.partitionBy("team_id").orderBy("game_date").rowsBetween(-4, 0)
        
        team_features = games_df.groupBy("game_id", "team_id", "game_date") \
            .agg(
                F.sum("points").alias("team_points"),
                F.sum("rebounds").alias("team_rebounds"),
                F.sum("assists").alias("team_assists"),
                F.avg("field_goal_percentage").alias("team_fg_pct"),
                F.sum("turnovers").alias("team_turnovers")
            ) \
            .withColumn("offensive_rating", 
                F.col("team_points") * 100 / F.lit(100))  # Simplified, would need possessions
            .withColumn("assist_ratio", 
                F.col("team_assists") / (F.col("team_fg_pct") * 100)) \
            .withColumn("avg_team_points_last_5", 
                F.avg("team_points").over(last_5_games)) \
            .withColumn("team_momentum", 
                F.col("team_points") - F.col("avg_team_points_last_5"))
        
        return team_features
    
    def compute_matchup_features(self, player_df: DataFrame, team_df: DataFrame) -> DataFrame:
        """Compute player vs team matchup features"""
        # Historical performance against specific teams
        matchup_window = Window.partitionBy("player_id", "opponent_team_id")
        
        matchup_features = player_df \
            .withColumn("avg_points_vs_team", 
                F.avg("points").over(matchup_window)) \
            .withColumn("games_vs_team", 
                F.count("game_id").over(matchup_window)) \
            .withColumn("last_performance_vs_team", 
                F.lag("points", 1).over(matchup_window.orderBy("game_date"))) \
            .withColumn("career_high_vs_team", 
                F.max("points").over(matchup_window))
        
        return matchup_features
    
    def create_feature_pipeline(self, raw_data_path: str) -> Pipeline:
        """Create full feature engineering pipeline"""
        # Read raw data
        games_df = self.spark.read.parquet(raw_data_path)
        
        # Compute all feature sets
        player_features = self.compute_player_features(games_df)
        team_features = self.compute_team_features(games_df)
        matchup_features = self.compute_matchup_features(player_features, team_features)
        
        # Select final features for modeling
        feature_columns = [
            "rest_days", "is_back_to_back", "games_played_season",
            "avg_points_last_5", "avg_points_last_10", "std_points_last_10",
            "avg_rebounds_last_5", "avg_assists_last_5", "avg_minutes_last_5",
            "shooting_pct_last_5", "usage_rate_last_5", "efficiency_rating_last_5",
            "points_momentum", "hot_streak", "consistency_score",
            "opponent_defensive_rating", "expected_points_vs_opponent",
            "fatigue_index", "avg_points_vs_team", "games_vs_team"
        ]
        
        # Create vector assembler
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        
        return pipeline, matchup_features
    
    def save_features_to_delta(self, features_df: DataFrame, path: str, 
                              partition_cols: List[str] = ["game_date"]) -> None:
        """Save features to Delta Lake for versioning and time travel"""
        features_df.write \
            .mode("overwrite") \
            .partitionBy(*partition_cols) \
            .format("delta") \
            .option("overwriteSchema", "true") \
            .save(path)
        
        # Optimize for better query performance
        delta_table = DeltaTable.forPath(self.spark, path)
        delta_table.optimize().executeCompaction()
        
        # Enable change data feed for downstream consumers
        self.spark.sql(f"ALTER TABLE delta.`{path}` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        
        logger.info(f"Saved {features_df.count()} feature rows to {path}")
    
    def incremental_feature_update(self, delta_path: str, new_data_df: DataFrame) -> None:
        """Incrementally update features in Delta Lake"""
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        
        # Merge new data
        delta_table.alias("existing") \
            .merge(
                new_data_df.alias("new"),
                "existing.player_id = new.player_id AND existing.game_date = new.game_date"
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()
        
        # Run optimize periodically
        delta_table.optimize().executeCompaction()
        
        # Vacuum old files (keep 7 days of history)
        delta_table.vacuum(168)


# Example usage
if __name__ == "__main__":
    spark_config = {
        "spark.executor.memory": "4g",
        "spark.executor.cores": "4",
        "spark.sql.shuffle.partitions": "200",
        "spark.default.parallelism": "100"
    }
    
    computer = SparkFeatureComputer(spark_config)
    
    # Create and run pipeline
    pipeline, features = computer.create_feature_pipeline("s3://nba-data/games/*.parquet")
    
    # Save to Delta Lake
    computer.save_features_to_delta(features, "s3://nba-features/delta/player_features")