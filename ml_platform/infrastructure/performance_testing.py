"""
Comprehensive Performance Testing Suite with Load Testing,
Chaos Engineering, and Performance Regression Detection
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import aiohttp
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
import requests
import random
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import docker
import kubernetes
from chaos_engineering import ChaosMonkey
import json

# Metrics
test_requests = Counter('test_requests_total', 'Test requests', ['test_type', 'status'])
test_latency = Histogram('test_latency_seconds', 'Test latency', ['test_type'])
test_throughput = Gauge('test_throughput_rps', 'Test throughput', ['test_type'])
resource_usage = Gauge('test_resource_usage', 'Resource usage during tests', ['resource_type'])

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    timestamp: datetime


@dataclass
class PerformanceBaseline:
    test_name: str
    avg_response_time: float
    p95_response_time: float
    throughput_rps: float
    error_rate: float
    resource_usage: Dict[str, float]


class NBAAnalyticsUser(HttpUser):
    """Locust user class for NBA Analytics API testing"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user session"""
        # Authenticate and get token
        response = self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
        else:
            logger.error("Failed to authenticate test user")
    
    @task(3)
    def get_player(self):
        """Test player retrieval"""
        player_ids = ["player_123", "player_456", "player_789"]
        player_id = random.choice(player_ids)
        
        with self.client.get(f"/graphql", 
                           json={
                               "query": f"""
                                   query {{
                                       player(id: "{player_id}") {{
                                           id
                                           name
                                           team
                                           currentSeasonStats {{
                                               pointsPerGame
                                               reboundsPerGame
                                               assistsPerGame
                                           }}
                                       }}
                                   }}
                               """
                           },
                           catch_response=True) as response:
            
            if response.status_code == 200:
                data = response.json()
                if "errors" in data:
                    response.failure(f"GraphQL errors: {data['errors']}")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def create_prediction(self):
        """Test prediction creation"""
        player_id = f"player_{random.randint(1, 1000)}"
        
        with self.client.post("/graphql",
                            json={
                                "query": """
                                    mutation CreatePrediction($input: PredictionInput!) {
                                        createPrediction(input: $input) {
                                            id
                                            pointsPrediction
                                            reboundsPrediction
                                            assistsPrediction
                                        }
                                    }
                                """,
                                "variables": {
                                    "input": {
                                        "playerId": player_id,
                                        "gameDate": "2024-01-15T19:00:00Z",
                                        "features": {
                                            "rest_days": random.uniform(0, 5),
                                            "opponent_rating": random.uniform(0.3, 1.0),
                                            "home_game": random.choice([0, 1])
                                        }
                                    }
                                }
                            },
                            catch_response=True) as response:
            
            if response.status_code == 200:
                data = response.json()
                if "errors" in data:
                    response.failure(f"GraphQL errors: {data['errors']}")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def bulk_predictions(self):
        """Test bulk prediction creation"""
        predictions = []
        for i in range(5):
            predictions.append({
                "playerId": f"player_{random.randint(1, 1000)}",
                "gameDate": "2024-01-15T19:00:00Z",
                "features": {
                    "rest_days": random.uniform(0, 5),
                    "opponent_rating": random.uniform(0.3, 1.0),
                    "home_game": random.choice([0, 1])
                }
            })
        
        with self.client.post("/graphql",
                            json={
                                "query": """
                                    mutation BulkPredictions($input: BulkPredictionInput!) {
                                        bulkPredictions(input: $input) {
                                            id
                                            pointsPrediction
                                        }
                                    }
                                """,
                                "variables": {
                                    "input": {
                                        "predictions": predictions
                                    }
                                }
                            },
                            catch_response=True) as response:
            
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class PerformanceTestSuite:
    """Comprehensive performance testing suite"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config['base_url']
        self.results = []
        self.baselines = self._load_baselines()
        
        # Docker client for chaos testing
        self.docker_client = docker.from_env()
        
        # Kubernetes client if available
        try:
            kubernetes.config.load_incluster_config()
            self.k8s_client = kubernetes.client.ApiClient()
        except:
            try:
                kubernetes.config.load_kube_config()
                self.k8s_client = kubernetes.client.ApiClient()
            except:
                self.k8s_client = None
                logger.warning("Kubernetes client not available")
    
    async def run_load_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Run load test with Locust"""
        logger.info(f"Starting load test: {test_config['name']}")
        
        # Setup Locust environment
        env = Environment(user_classes=[NBAAnalyticsUser])
        env.create_local_runner()
        
        # Configure test parameters
        user_count = test_config.get('users', 10)
        spawn_rate = test_config.get('spawn_rate', 2)
        run_time = test_config.get('run_time', 60)
        
        # Start test
        start_time = time.time()
        env.runner.start(user_count=user_count, spawn_rate=spawn_rate)
        
        # Run for specified duration
        await asyncio.sleep(run_time)
        
        # Stop test
        env.runner.stop()
        end_time = time.time()
        
        # Collect results
        stats = env.runner.stats
        total = stats.total
        
        result = TestResult(
            test_name=test_config['name'],
            duration_seconds=end_time - start_time,
            total_requests=total.num_requests,
            successful_requests=total.num_requests - total.num_failures,
            failed_requests=total.num_failures,
            avg_response_time=total.avg_response_time,
            p50_response_time=total.get_response_time_percentile(0.5),
            p95_response_time=total.get_response_time_percentile(0.95),
            p99_response_time=total.get_response_time_percentile(0.99),
            throughput_rps=total.total_rps,
            error_rate=total.num_failures / max(total.num_requests, 1),
            timestamp=datetime.utcnow()
        )
        
        # Update metrics
        test_requests.labels(
            test_type=test_config['name'],
            status='completed'
        ).inc(total.num_requests)
        
        test_latency.labels(test_type=test_config['name']).observe(
            total.avg_response_time / 1000
        )
        
        test_throughput.labels(test_type=test_config['name']).set(total.total_rps)
        
        self.results.append(result)
        logger.info(f"Load test completed: {result}")
        
        return result
    
    async def run_stress_test(self, max_users: int = 1000, 
                            ramp_duration: int = 300) -> List[TestResult]:
        """Run stress test with gradual load increase"""
        logger.info(f"Starting stress test up to {max_users} users")
        
        results = []
        user_increments = [10, 25, 50, 100, 200, 500, max_users]
        
        for user_count in user_increments:
            if user_count > max_users:
                break
                
            logger.info(f"Testing with {user_count} users")
            
            test_config = {
                'name': f'stress_test_{user_count}_users',
                'users': user_count,
                'spawn_rate': min(user_count // 10, 20),
                'run_time': 120  # 2 minutes per level
            }
            
            result = await self.run_load_test(test_config)
            results.append(result)
            
            # Check if system is degrading
            if result.error_rate > 0.1 or result.p95_response_time > 5000:
                logger.warning(f"System degradation detected at {user_count} users")
                break
            
            # Cool down period
            await asyncio.sleep(30)
        
        return results
    
    async def run_endurance_test(self, duration_hours: int = 2) -> TestResult:
        """Run endurance test for extended period"""
        logger.info(f"Starting endurance test for {duration_hours} hours")
        
        test_config = {
            'name': f'endurance_test_{duration_hours}h',
            'users': 50,
            'spawn_rate': 5,
            'run_time': duration_hours * 3600
        }
        
        # Monitor resource usage during test
        resource_monitor = asyncio.create_task(
            self._monitor_resources(duration_hours * 3600)
        )
        
        # Run test
        result = await self.run_load_test(test_config)
        
        # Stop resource monitoring
        resource_monitor.cancel()
        
        return result
    
    async def run_spike_test(self, base_users: int = 10, 
                           spike_users: int = 100) -> TestResult:
        """Run spike test with sudden load increase"""
        logger.info(f"Starting spike test: {base_users} -> {spike_users} users")
        
        # Start with base load
        env = Environment(user_classes=[NBAAnalyticsUser])
        env.create_local_runner()
        
        start_time = time.time()
        
        # Base load for 2 minutes
        env.runner.start(user_count=base_users, spawn_rate=5)
        await asyncio.sleep(120)
        
        # Sudden spike
        env.runner.start(user_count=spike_users, spawn_rate=spike_users)
        await asyncio.sleep(300)  # 5 minutes of spike
        
        # Back to base load
        env.runner.start(user_count=base_users, spawn_rate=10)
        await asyncio.sleep(120)
        
        env.runner.stop()
        end_time = time.time()
        
        # Collect results
        stats = env.runner.stats.total
        
        result = TestResult(
            test_name='spike_test',
            duration_seconds=end_time - start_time,
            total_requests=stats.num_requests,
            successful_requests=stats.num_requests - stats.num_failures,
            failed_requests=stats.num_failures,
            avg_response_time=stats.avg_response_time,
            p50_response_time=stats.get_response_time_percentile(0.5),
            p95_response_time=stats.get_response_time_percentile(0.95),
            p99_response_time=stats.get_response_time_percentile(0.99),
            throughput_rps=stats.total_rps,
            error_rate=stats.num_failures / max(stats.num_requests, 1),
            timestamp=datetime.utcnow()
        )
        
        return result
    
    async def run_chaos_test(self, duration_minutes: int = 30) -> TestResult:
        """Run chaos engineering test"""
        logger.info(f"Starting chaos test for {duration_minutes} minutes")
        
        # Start baseline load
        test_config = {
            'name': 'chaos_test',
            'users': 25,
            'spawn_rate': 5,
            'run_time': duration_minutes * 60
        }
        
        # Start load test
        load_test_task = asyncio.create_task(self.run_load_test(test_config))
        
        # Start chaos monkey
        chaos_monkey = ChaosMonkey(self.docker_client, self.k8s_client)
        
        chaos_scenarios = [
            ('kill_random_container', 0.3),
            ('network_delay', 0.2),
            ('cpu_stress', 0.2),
            ('memory_stress', 0.2),
            ('disk_stress', 0.1)
        ]
        
        # Run chaos scenarios during load test
        for scenario, probability in chaos_scenarios:
            if random.random() < probability:
                logger.info(f"Executing chaos scenario: {scenario}")
                await chaos_monkey.execute_scenario(scenario)
                await asyncio.sleep(random.randint(60, 180))  # Random interval
        
        # Wait for load test to complete
        result = await load_test_task
        
        return result
    
    async def compare_with_baseline(self, result: TestResult) -> Dict[str, Any]:
        """Compare test result with baseline"""
        baseline = self.baselines.get(result.test_name)
        
        if not baseline:
            logger.warning(f"No baseline found for {result.test_name}")
            return {'status': 'no_baseline'}
        
        comparison = {
            'test_name': result.test_name,
            'status': 'pass',
            'degradations': [],
            'improvements': [],
            'metrics_comparison': {}
        }
        
        # Compare key metrics
        metrics = [
            ('avg_response_time', 'lower_is_better', 0.2),  # 20% degradation threshold
            ('p95_response_time', 'lower_is_better', 0.25),
            ('throughput_rps', 'higher_is_better', 0.15),
            ('error_rate', 'lower_is_better', 0.1)
        ]
        
        for metric, direction, threshold in metrics:
            current_value = getattr(result, metric)
            baseline_value = getattr(baseline, metric)
            
            if baseline_value == 0:
                continue
                
            change_ratio = (current_value - baseline_value) / baseline_value
            
            comparison['metrics_comparison'][metric] = {
                'current': current_value,
                'baseline': baseline_value,
                'change_ratio': change_ratio,
                'change_percent': change_ratio * 100
            }
            
            # Check for degradation/improvement
            if direction == 'lower_is_better':
                if change_ratio > threshold:
                    comparison['degradations'].append({
                        'metric': metric,
                        'change_percent': change_ratio * 100,
                        'threshold_percent': threshold * 100
                    })
                    comparison['status'] = 'degraded'
                elif change_ratio < -0.1:  # 10% improvement
                    comparison['improvements'].append({
                        'metric': metric,
                        'improvement_percent': abs(change_ratio * 100)
                    })
            else:  # higher_is_better
                if change_ratio < -threshold:
                    comparison['degradations'].append({
                        'metric': metric,
                        'change_percent': change_ratio * 100,
                        'threshold_percent': threshold * 100
                    })
                    comparison['status'] = 'degraded'
                elif change_ratio > 0.1:  # 10% improvement
                    comparison['improvements'].append({
                        'metric': metric,
                        'improvement_percent': change_ratio * 100
                    })
        
        return comparison
    
    async def generate_performance_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not results:
            return {'error': 'No test results to analyze'}
        
        # Calculate summary statistics
        summary = {
            'total_tests': len(results),
            'total_requests': sum(r.total_requests for r in results),
            'total_failures': sum(r.failed_requests for r in results),
            'avg_throughput': np.mean([r.throughput_rps for r in results]),
            'avg_error_rate': np.mean([r.error_rate for r in results]),
            'test_duration': sum(r.duration_seconds for r in results),
            'test_period': {
                'start': min(r.timestamp for r in results),
                'end': max(r.timestamp for r in results)
            }
        }
        
        # Performance trends
        trends = self._analyze_performance_trends(results)
        
        # Baseline comparisons
        comparisons = []
        for result in results:
            comparison = await self.compare_with_baseline(result)
            if comparison['status'] != 'no_baseline':
                comparisons.append(comparison)
        
        # Resource utilization
        resource_analysis = self._analyze_resource_usage()
        
        # Recommendations
        recommendations = self._generate_recommendations(results, comparisons)
        
        report = {
            'summary': summary,
            'trends': trends,
            'baseline_comparisons': comparisons,
            'resource_analysis': resource_analysis,
            'recommendations': recommendations,
            'detailed_results': [asdict(r) for r in results],
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Save report
        report_path = f"performance_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {report_path}")
        
        return report
    
    async def _monitor_resources(self, duration_seconds: int):
        """Monitor system resources during test"""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                resource_usage.labels(resource_type='cpu').set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                resource_usage.labels(resource_type='memory').set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                resource_usage.labels(resource_type='disk').set(disk_percent)
                
                # Network I/O
                network = psutil.net_io_counters()
                resource_usage.labels(resource_type='network_in').set(network.bytes_recv)
                resource_usage.labels(resource_type='network_out').set(network.bytes_sent)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)
    
    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from storage"""
        try:
            with open('performance_baselines.json', 'r') as f:
                baselines_data = json.load(f)
                
            baselines = {}
            for name, data in baselines_data.items():
                baselines[name] = PerformanceBaseline(**data)
                
            return baselines
        except FileNotFoundError:
            logger.warning("No baseline file found, creating empty baselines")
            return {}
        except Exception as e:
            logger.error(f"Error loading baselines: {e}")
            return {}
    
    def _analyze_performance_trends(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(results) < 2:
            return {'insufficient_data': True}
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x.timestamp)
        
        # Calculate trends
        timestamps = [r.timestamp for r in sorted_results]
        response_times = [r.avg_response_time for r in sorted_results]
        throughput = [r.throughput_rps for r in sorted_results]
        error_rates = [r.error_rate for r in sorted_results]
        
        trends = {
            'response_time_trend': np.polyfit(range(len(response_times)), response_times, 1)[0],
            'throughput_trend': np.polyfit(range(len(throughput)), throughput, 1)[0],
            'error_rate_trend': np.polyfit(range(len(error_rates)), error_rates, 1)[0],
            'performance_score': self._calculate_performance_score(sorted_results[-1])
        }
        
        return trends
    
    def _calculate_performance_score(self, result: TestResult) -> float:
        """Calculate overall performance score (0-100)"""
        # Weighted scoring based on key metrics
        throughput_score = min(result.throughput_rps / 100, 1.0) * 30  # Max 30 points
        latency_score = max(0, (2000 - result.p95_response_time) / 2000) * 40  # Max 40 points
        reliability_score = (1 - result.error_rate) * 30  # Max 30 points
        
        return throughput_score + latency_score + reliability_score
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        return {
            'cpu_analysis': 'Resource analysis would be implemented here',
            'memory_analysis': 'Memory usage patterns',
            'network_analysis': 'Network I/O analysis',
            'bottlenecks': []
        }
    
    def _generate_recommendations(self, results: List[TestResult], 
                                comparisons: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze latest result
        if results:
            latest = results[-1]
            
            if latest.error_rate > 0.05:
                recommendations.append(
                    f"High error rate ({latest.error_rate:.1%}) detected. "
                    "Investigate application logs and error handling."
                )
            
            if latest.p95_response_time > 2000:
                recommendations.append(
                    f"High P95 latency ({latest.p95_response_time:.0f}ms). "
                    "Consider optimizing database queries and caching."
                )
            
            if latest.throughput_rps < 50:
                recommendations.append(
                    f"Low throughput ({latest.throughput_rps:.1f} RPS). "
                    "Consider scaling up or optimizing application performance."
                )
        
        # Analyze degradations
        degraded_tests = [c for c in comparisons if c['status'] == 'degraded']
        if degraded_tests:
            recommendations.append(
                f"Performance degradation detected in {len(degraded_tests)} tests. "
                "Review recent changes and consider rollback if necessary."
            )
        
        return recommendations


class ChaosMonkey:
    """Chaos engineering implementation"""
    
    def __init__(self, docker_client, k8s_client):
        self.docker_client = docker_client
        self.k8s_client = k8s_client
    
    async def execute_scenario(self, scenario: str):
        """Execute chaos engineering scenario"""
        scenarios = {
            'kill_random_container': self._kill_random_container,
            'network_delay': self._introduce_network_delay,
            'cpu_stress': self._cpu_stress_test,
            'memory_stress': self._memory_stress_test,
            'disk_stress': self._disk_stress_test
        }
        
        if scenario in scenarios:
            await scenarios[scenario]()
        else:
            logger.warning(f"Unknown chaos scenario: {scenario}")
    
    async def _kill_random_container(self):
        """Kill a random container"""
        try:
            containers = self.docker_client.containers.list()
            if containers:
                container = random.choice(containers)
                logger.info(f"Killing container: {container.name}")
                container.kill()
                await asyncio.sleep(5)  # Wait for restart
        except Exception as e:
            logger.error(f"Failed to kill container: {e}")
    
    async def _introduce_network_delay(self):
        """Introduce network latency"""
        # This would use tools like tc (traffic control) or toxiproxy
        logger.info("Introducing network delay (mock)")
        await asyncio.sleep(60)  # Mock delay
    
    async def _cpu_stress_test(self):
        """Create CPU stress"""
        logger.info("Starting CPU stress test")
        # Run stress test container
        try:
            self.docker_client.containers.run(
                "progrium/stress",
                "--cpu 2 --timeout 60s",
                detach=True,
                remove=True
            )
        except Exception as e:
            logger.error(f"CPU stress test failed: {e}")
    
    async def _memory_stress_test(self):
        """Create memory pressure"""
        logger.info("Starting memory stress test")
        # Implementation would use stress-ng or similar
        await asyncio.sleep(60)
    
    async def _disk_stress_test(self):
        """Create disk I/O stress"""
        logger.info("Starting disk stress test")
        # Implementation would use fio or dd
        await asyncio.sleep(60)


async def main():
    """Main function to run performance tests"""
    config = {
        'base_url': 'http://localhost:8000',
        'auth_token': 'your-test-token'
    }
    
    test_suite = PerformanceTestSuite(config)
    
    # Run different types of tests
    tests_to_run = [
        {'name': 'baseline_load', 'users': 10, 'run_time': 300},
        {'name': 'medium_load', 'users': 50, 'run_time': 300},
        {'name': 'high_load', 'users': 100, 'run_time': 300}
    ]
    
    results = []
    for test_config in tests_to_run:
        result = await test_suite.run_load_test(test_config)
        results.append(result)
    
    # Run stress test
    stress_results = await test_suite.run_stress_test(max_users=200)
    results.extend(stress_results)
    
    # Run spike test
    spike_result = await test_suite.run_spike_test()
    results.append(spike_result)
    
    # Generate report
    report = await test_suite.generate_performance_report(results)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    # Setup logging
    setup_logging("INFO", None)
    
    # Start metrics server
    start_http_server(8001)
    
    # Run tests
    asyncio.run(main())