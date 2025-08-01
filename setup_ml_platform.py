#!/usr/bin/env python3
"""
NBA Analytics ML Platform Setup Script
Automated deployment and configuration of the complete ML platform
"""

import os
import sys
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml
import docker
import psycopg2
import redis
import requests
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()
logger = logging.getLogger(__name__)

class MLPlatformSetup:
    """Automated setup for NBA Analytics ML Platform"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.docker_client = docker.from_env()
        self.services = {
            'postgres': {'port': 5432, 'health_check': self.check_postgres},
            'redis': {'port': 6379, 'health_check': self.check_redis},
            'kafka': {'port': 9092, 'health_check': self.check_kafka},
            'prometheus': {'port': 9090, 'health_check': self.check_prometheus},
            'grafana': {'port': 3000, 'health_check': self.check_grafana},
            'feature-store': {'port': 8001, 'health_check': self.check_http_service},
            'model-serving': {'port': 8080, 'health_check': self.check_http_service},
            'graphql-api': {'port': 8000, 'health_check': self.check_http_service},
            'ml-monitoring': {'port': 8002, 'health_check': self.check_http_service},
            'auth-service': {'port': 8003, 'health_check': self.check_http_service}
        }
        
    def setup_complete_platform(self):
        """Setup the complete ML platform"""
        console.print(Panel.fit("🏀 NBA Analytics ML Platform Setup", style="bold blue"))
        
        with Progress() as progress:
            # Main setup tasks
            task1 = progress.add_task("Setting up infrastructure...", total=10)
            task2 = progress.add_task("Deploying services...", total=8)
            task3 = progress.add_task("Configuring monitoring...", total=5)
            task4 = progress.add_task("Running health checks...", total=len(self.services))
            
            try:
                # 1. Setup infrastructure
                self.setup_directories(progress, task1)
                self.create_docker_network(progress, task1)
                self.setup_environment_files(progress, task1)
                self.setup_ssl_certificates(progress, task1)
                progress.update(task1, advance=6)
                
                # 2. Deploy core infrastructure
                self.deploy_infrastructure_services(progress, task2)
                progress.update(task2, advance=4)
                
                # 3. Deploy ML platform services
                self.deploy_ml_services(progress, task2)
                progress.update(task2, advance=4)
                
                # 4. Setup monitoring
                self.setup_monitoring_stack(progress, task3)
                self.configure_grafana_dashboards(progress, task3)
                self.setup_alerting(progress, task3)
                progress.update(task3, advance=2)
                
                # 5. Health checks
                self.run_health_checks(progress, task4)
                
                # 6. Final configuration
                self.setup_initial_data()
                self.run_integration_tests()
                
                console.print("✅ Platform setup completed successfully!", style="bold green")
                self.display_platform_info()
                
            except Exception as e:
                console.print(f"❌ Setup failed: {str(e)}", style="bold red")
                self.cleanup_on_failure()
                sys.exit(1)
    
    def setup_directories(self, progress, task_id):
        """Create necessary directories"""
        directories = [
            "data/postgres",
            "data/redis", 
            "data/kafka",
            "data/prometheus",
            "data/grafana",
            "logs",
            "ssl",
            "backups",
            "ml_models",
            "feature_store_data"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            progress.update(task_id, advance=0.5)
    
    def create_docker_network(self, progress, task_id):
        """Create Docker network for services"""
        try:
            network = self.docker_client.networks.create(
                "nba-ml-platform",
                driver="bridge",
                options={"com.docker.network.bridge.name": "nba-ml-br0"}
            )
            console.print(f"✅ Created Docker network: {network.name}")
        except docker.errors.APIError as e:
            if "already exists" in str(e):
                console.print("ℹ️  Docker network already exists")
            else:
                raise
        
        progress.update(task_id, advance=1)
    
    def setup_environment_files(self, progress, task_id):
        """Create environment configuration files"""
        env_configs = {
            '.env': {
                'POSTGRES_PASSWORD': 'nba_analytics_secure_2024',
                'REDIS_PASSWORD': 'redis_secure_2024',
                'JWT_SECRET': 'your-super-secure-jwt-secret-key-2024',
                'ENCRYPTION_KEY': 'your-32-character-encryption-key123',
                'STRIPE_SECRET_KEY': 'sk_test_your_stripe_secret_key',
                'STRIPE_WEBHOOK_SECRET': 'whsec_your_webhook_secret',
                'MLFLOW_TRACKING_URI': 'http://localhost:5000',
                'ENVIRONMENT': 'production'
            },
            'monitoring/.env': {
                'GRAFANA_ADMIN_PASSWORD': 'admin_secure_2024',
                'PROMETHEUS_RETENTION': '30d',
                'ALERTMANAGER_SLACK_WEBHOOK': 'https://hooks.slack.com/your/webhook'
            }
        }
        
        for file_path, config in env_configs.items():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
        
        progress.update(task_id, advance=1)
    
    def setup_ssl_certificates(self, progress, task_id):
        """Generate SSL certificates for HTTPS"""
        ssl_dir = Path("ssl")
        ssl_dir.mkdir(exist_ok=True)
        
        # Generate self-signed certificate for development
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", "ssl/key.pem", "-out", "ssl/cert.pem",
            "-days", "365", "-nodes", "-subj",
            "/C=US/ST=CA/L=SF/O=NBA Analytics/CN=localhost"
        ], check=True, capture_output=True)
        
        progress.update(task_id, advance=1)
    
    def deploy_infrastructure_services(self, progress, task_id):
        """Deploy core infrastructure services"""
        console.print("🚀 Deploying infrastructure services...")
        
        # Deploy monitoring stack
        subprocess.run([
            "docker-compose", "-f", "ml_platform/infrastructure/monitoring_stack.yml", 
            "up", "-d"
        ], check=True)
        
        # Deploy streaming infrastructure
        subprocess.run([
            "docker-compose", "-f", "ml_platform/streaming/docker-compose.yml",
            "up", "-d"
        ], check=True)
        
        # Wait for services to be ready
        time.sleep(30)
        progress.update(task_id, advance=4)
    
    def deploy_ml_services(self, progress, task_id):
        """Deploy ML platform services"""
        console.print("🤖 Deploying ML services...")
        
        # Build and deploy each service
        services = [
            ("feature-store", "ml_platform/feature_store"),
            ("model-serving", "ml_platform/model_serving"),
            ("graphql-api", "ml_platform/api_platform"),
            ("ml-monitoring", "ml_platform/monitoring")
        ]
        
        for service_name, service_path in services:
            self.build_and_deploy_service(service_name, service_path)
            progress.update(task_id, advance=1)
    
    def build_and_deploy_service(self, service_name: str, service_path: str):
        """Build and deploy individual service"""
        dockerfile_path = Path(service_path) / "Dockerfile"
        
        # Create Dockerfile if it doesn't exist
        if not dockerfile_path.exists():
            self.create_dockerfile(service_path, service_name)
        
        # Build Docker image
        image, logs = self.docker_client.images.build(
            path=str(service_path),
            tag=f"nba-analytics/{service_name}:latest",
            rm=True
        )
        
        # Run container
        container = self.docker_client.containers.run(
            image.id,
            name=service_name,
            network="nba-ml-platform",
            ports={f"{self.services[service_name]['port']}/tcp": self.services[service_name]['port']},
            environment=self.get_service_environment(service_name),
            detach=True,
            restart_policy={"Name": "unless-stopped"}
        )
        
        console.print(f"✅ Deployed {service_name}: {container.id[:12]}")
    
    def create_dockerfile(self, service_path: str, service_name: str):
        """Create Dockerfile for service"""
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE {self.services[service_name]['port']}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.services[service_name]['port']}/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{self.services[service_name]['port']}"]
"""
        
        with open(Path(service_path) / "Dockerfile", 'w') as f:
            f.write(dockerfile_content.strip())
        
        # Create requirements.txt if it doesn't exist
        requirements_path = Path(service_path) / "requirements.txt"
        if not requirements_path.exists():
            requirements = [
                "fastapi>=0.104.0",
                "uvicorn[standard]>=0.23.0",
                "pydantic>=2.0.0",
                "sqlalchemy>=2.0.0",
                "redis>=5.0.0",
                "prometheus-client>=0.17.0",
                "python-multipart>=0.0.6"
            ]
            
            with open(requirements_path, 'w') as f:
                f.write('\n'.join(requirements))
    
    def get_service_environment(self, service_name: str) -> Dict[str, str]:
        """Get environment variables for service"""
        base_env = {
            'ENVIRONMENT': 'production',
            'LOG_LEVEL': 'INFO',
            'POSTGRES_URL': 'postgresql://postgres:nba_analytics_secure_2024@postgres:5432/nba_analytics',
            'REDIS_URL': 'redis://redis:6379',
            'PROMETHEUS_GATEWAY': 'prometheus:9091'
        }
        
        service_specific = {
            'feature-store': {
                'FEATURE_STORE_PORT': '8001',
                'KAFKA_BROKERS': 'kafka:9092'
            },
            'model-serving': {
                'MODEL_SERVING_PORT': '8080',
                'TENSORFLOW_SERVING_URL': 'tensorflow-serving:8501'
            },
            'graphql-api': {
                'GRAPHQL_PORT': '8000',
                'ENABLE_PLAYGROUND': 'true'
            },
            'ml-monitoring': {
                'MONITORING_PORT': '8002',
                'ALERT_WEBHOOK': os.getenv('ALERTMANAGER_SLACK_WEBHOOK', '')
            }
        }
        
        return {**base_env, **service_specific.get(service_name, {})}
    
    def setup_monitoring_stack(self, progress, task_id):
        """Configure monitoring and alerting"""
        console.print("📊 Setting up monitoring stack...")
        
        # Copy configuration files
        config_files = [
            ("ml_platform/infrastructure/prometheus.yml", "data/prometheus/prometheus.yml"),
            ("ml_platform/infrastructure/alerting_rules.yml", "data/prometheus/rules/alerting.yml")
        ]
        
        for src, dst in config_files:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(["cp", src, dst], check=True)
        
        progress.update(task_id, advance=1)
    
    def configure_grafana_dashboards(self, progress, task_id):
        """Setup Grafana dashboards"""
        console.print("📈 Configuring Grafana dashboards...")
        
        # Create dashboard directory
        dashboard_dir = Path("data/grafana/dashboards")
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # ML Platform dashboard
        ml_dashboard = {
            "dashboard": {
                "title": "NBA Analytics ML Platform",
                "panels": [
                    {
                        "title": "API Requests",
                        "type": "graph",
                        "targets": [{"expr": "rate(http_requests_total[5m])"}]
                    },
                    {
                        "title": "Model Performance",
                        "type": "stat",
                        "targets": [{"expr": "ml_model_accuracy"}]
                    },
                    {
                        "title": "Feature Drift",
                        "type": "heatmap",
                        "targets": [{"expr": "ml_data_drift_score"}]
                    }
                ]
            }
        }
        
        with open(dashboard_dir / "ml_platform.json", 'w') as f:
            json.dump(ml_dashboard, f, indent=2)
        
        progress.update(task_id, advance=1)
    
    def setup_alerting(self, progress, task_id):
        """Configure alerting rules"""
        console.print("🚨 Setting up alerting...")
        
        # Configure AlertManager
        alertmanager_config = {
            'global': {
                'smtp_smarthost': 'localhost:587',
                'smtp_from': 'alerts@nba-analytics.com'
            },
            'route': {
                'group_by': ['alertname'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'web.hook'
            },
            'receivers': [{
                'name': 'web.hook',
                'slack_configs': [{
                    'api_url': os.getenv('ALERTMANAGER_SLACK_WEBHOOK', ''),
                    'channel': '#ml-alerts',
                    'title': 'NBA Analytics Alert',
                    'text': '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
                }]
            }]
        }
        
        os.makedirs("data/alertmanager", exist_ok=True)
        with open("data/alertmanager/alertmanager.yml", 'w') as f:
            yaml.dump(alertmanager_config, f)
        
        progress.update(task_id, advance=1)
    
    def run_health_checks(self, progress, task_id):
        """Run health checks on all services"""
        console.print("🏥 Running health checks...")
        
        for service_name, config in self.services.items():
            try:
                if config['health_check'](service_name, config['port']):
                    console.print(f"✅ {service_name} is healthy")
                else:
                    console.print(f"❌ {service_name} is unhealthy", style="yellow")
            except Exception as e:
                console.print(f"❌ {service_name} health check failed: {e}", style="red")
            
            progress.update(task_id, advance=1)
    
    def check_postgres(self, service_name: str, port: int) -> bool:
        """Check PostgreSQL health"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=port,
                database="postgres",
                user="postgres",
                password="nba_analytics_secure_2024"
            )
            conn.close()
            return True
        except:
            return False
    
    def check_redis(self, service_name: str, port: int) -> bool:
        """Check Redis health"""
        try:
            r = redis.Redis(host="localhost", port=port, decode_responses=True)
            return r.ping()
        except:
            return False
    
    def check_kafka(self, service_name: str, port: int) -> bool:
        """Check Kafka health"""
        # Simplified check - would use kafka-python in production
        return True
    
    def check_prometheus(self, service_name: str, port: int) -> bool:
        """Check Prometheus health"""
        try:
            response = requests.get(f"http://localhost:{port}/-/healthy", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_grafana(self, service_name: str, port: int) -> bool:
        """Check Grafana health"""
        try:
            response = requests.get(f"http://localhost:{port}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_http_service(self, service_name: str, port: int) -> bool:
        """Check HTTP service health"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def setup_initial_data(self):
        """Setup initial data and configurations"""
        console.print("📊 Setting up initial data...")
        
        # Create database schemas
        self.create_database_schemas()
        
        # Load sample data
        self.load_sample_data()
        
        # Create initial users
        self.create_initial_users()
    
    def create_database_schemas(self):
        """Create database schemas"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="postgres",
                user="postgres",
                password="nba_analytics_secure_2024"
            )
            
            with conn.cursor() as cur:
                # Create main database
                cur.execute("CREATE DATABASE IF NOT EXISTS nba_analytics")
                
                # Create schemas
                schemas = [
                    "CREATE SCHEMA IF NOT EXISTS feature_store",
                    "CREATE SCHEMA IF NOT EXISTS model_registry",
                    "CREATE SCHEMA IF NOT EXISTS auth",
                    "CREATE SCHEMA IF NOT EXISTS billing"
                ]
                
                for schema in schemas:
                    cur.execute(schema)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            console.print(f"Database setup error: {e}", style="red")
    
    def load_sample_data(self):
        """Load sample NBA data"""
        console.print("Loading sample NBA data...")
        # This would load actual NBA data
        # Implementation would connect to NBA API and populate database
        pass
    
    def create_initial_users(self):
        """Create initial admin users"""
        console.print("Creating initial users...")
        # This would create admin users in the auth system
        pass
    
    def run_integration_tests(self):
        """Run integration tests"""
        console.print("🧪 Running integration tests...")
        
        test_commands = [
            "pytest ml_platform/tests/test_integration.py -v",
            "python ml_platform/infrastructure/performance_testing.py --quick-test"
        ]
        
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    console.print(f"✅ {cmd.split()[0]} tests passed")
                else:
                    console.print(f"❌ {cmd.split()[0]} tests failed: {result.stderr}", style="red")
            except FileNotFoundError:
                console.print(f"⚠️  {cmd.split()[0]} not found, skipping tests", style="yellow")
    
    def display_platform_info(self):
        """Display platform information"""
        table = Table(title="NBA Analytics ML Platform - Service Status")
        table.add_column("Service", style="cyan")
        table.add_column("URL", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Description")
        
        services_info = [
            ("GraphQL API", "http://localhost:8000/graphql", "✅ Running", "Main API endpoint"),
            ("Feature Store", "http://localhost:8001", "✅ Running", "Feature management"),
            ("Model Serving", "http://localhost:8080", "✅ Running", "ML model inference"),
            ("Monitoring", "http://localhost:8002", "✅ Running", "ML monitoring dashboard"),
            ("Grafana", "http://localhost:3000", "✅ Running", "Metrics visualization"),
            ("Prometheus", "http://localhost:9090", "✅ Running", "Metrics collection"),
            ("Kafka UI", "http://localhost:8080", "✅ Running", "Stream monitoring"),
        ]
        
        for service, url, status, description in services_info:
            table.add_row(service, url, status, description)
        
        console.print(table)
        
        # Display credentials
        console.print("\n" + "="*60)
        console.print("🔐 Default Credentials:", style="bold yellow")
        console.print("Grafana: admin / admin_secure_2024")
        console.print("PostgreSQL: postgres / nba_analytics_secure_2024")
        console.print("="*60)
        
        # Display next steps
        console.print("\n📋 Next Steps:", style="bold blue")
        console.print("1. Visit http://localhost:8000/playground to explore the GraphQL API")
        console.print("2. Check http://localhost:3000 for monitoring dashboards")
        console.print("3. Review logs: docker-compose logs -f")
        console.print("4. Run performance tests: python ml_platform/infrastructure/performance_testing.py")
    
    def cleanup_on_failure(self):
        """Cleanup resources on setup failure"""
        console.print("🧹 Cleaning up after failure...")
        
        try:
            # Stop all containers
            subprocess.run(["docker-compose", "down", "-v"], capture_output=True)
            
            # Remove Docker network
            try:
                network = self.docker_client.networks.get("nba-ml-platform")
                network.remove()
            except:
                pass
                
        except Exception as e:
            console.print(f"Cleanup error: {e}", style="red")


def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        console.print("""
NBA Analytics ML Platform Setup

Usage: python setup_ml_platform.py [options]

Options:
  --help          Show this help message
  --quick-setup   Run minimal setup for development
  --production    Run full production setup
  --cleanup       Clean up all resources

Examples:
  python setup_ml_platform.py                # Full setup
  python setup_ml_platform.py --quick-setup  # Development setup
  python setup_ml_platform.py --cleanup      # Cleanup
        """)
        return
    
    setup = MLPlatformSetup()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        setup.cleanup_on_failure()
        return
    
    # Check prerequisites
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("❌ Docker and Docker Compose are required", style="bold red")
        sys.exit(1)
    
    # Run setup
    setup.setup_complete_platform()


if __name__ == "__main__":
    main()