import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: float = 5.0
    max_connections: int = 50


@dataclass
class ModelConfig:
    """Model service configuration."""
    model_path: str = "ml-models/exported"
    enable_shap: bool = False
    inference_timeout: float = 10.0
    max_batch_size: int = 100
    fallback_mode: bool = True


@dataclass
class APIConfig:
    """API service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    cors_origins: list = None
    enable_docs: bool = True
    api_key_required: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_keys: list = None
    rate_limit_per_minute: int = 1000
    enable_request_validation: bool = True
    secure_headers: bool = True
    trusted_proxies: list = None


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enable_prometheus: bool = True
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    slow_request_threshold: float = 0.1  # 100ms
    enable_detailed_logging: bool = False


class FraudDetectionConfig:
    """
    Centralized configuration management for the fraud detection system.
    
    Loads configuration from environment variables with sensible defaults
    for production deployment.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.redis = self._load_redis_config()
        self.model = self._load_model_config()
        self.api = self._load_api_config()
        self.logging = self._load_logging_config()
        self.security = self._load_security_config()
        self.monitoring = self._load_monitoring_config()
        
        # Validate configuration
        self._validate_config()
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment."""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            socket_timeout=float(os.getenv("REDIS_TIMEOUT", "5.0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        )
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment."""
        return ModelConfig(
            model_path=os.getenv("MODEL_PATH", "ml-models/exported"),
            enable_shap=os.getenv("ENABLE_SHAP", "false").lower() == "true",
            inference_timeout=float(os.getenv("MODEL_INFERENCE_TIMEOUT", "10.0")),
            max_batch_size=int(os.getenv("MODEL_MAX_BATCH_SIZE", "100")),
            fallback_mode=os.getenv("MODEL_FALLBACK_MODE", "true").lower() == "true"
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment."""
        cors_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
        
        return APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            max_request_size=int(os.getenv("API_MAX_REQUEST_SIZE", str(16 * 1024 * 1024))),
            cors_origins=cors_origins,
            enable_docs=os.getenv("API_ENABLE_DOCS", "true").lower() == "true",
            api_key_required=os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment."""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration from environment."""
        api_keys = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
        trusted_proxies = os.getenv("TRUSTED_PROXIES", "").split(",") if os.getenv("TRUSTED_PROXIES") else []
        
        return SecurityConfig(
            api_keys=api_keys,
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "1000")),
            enable_request_validation=os.getenv("ENABLE_REQUEST_VALIDATION", "true").lower() == "true",
            secure_headers=os.getenv("SECURE_HEADERS", "true").lower() == "true",
            trusted_proxies=trusted_proxies
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration from environment."""
        return MonitoringConfig(
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            metrics_path=os.getenv("METRICS_PATH", "/metrics"),
            health_check_path=os.getenv("HEALTH_CHECK_PATH", "/health"),
            slow_request_threshold=float(os.getenv("SLOW_REQUEST_THRESHOLD", "0.1")),
            enable_detailed_logging=os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
        )
    
    def _validate_config(self):
        """Validate configuration values."""
        errors = []
        
        # Validate Redis configuration
        if not self.redis.host:
            errors.append("Redis host cannot be empty")
        if not (1 <= self.redis.port <= 65535):
            errors.append("Redis port must be between 1 and 65535")
        
        # Validate API configuration
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        if self.api.workers < 1:
            errors.append("API workers must be at least 1")
        
        # Validate model configuration
        if not self.model.model_path:
            errors.append("Model path cannot be empty")
        
        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation errors: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for backward compatibility."""
        return {
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'password': self.redis.password,
                'ssl': self.redis.ssl,
                'socket_timeout': self.redis.socket_timeout,
                'max_connections': self.redis.max_connections
            },
            'model_path': self.model.model_path,
            'host': self.api.host,
            'port': self.api.port,
            'log_level': self.logging.level,
            'workers': self.api.workers,
            'cors_origins': self.api.cors_origins,
            'enable_docs': self.api.enable_docs,
            'api_key_required': self.api.api_key_required,
            'rate_limit_per_minute': self.security.rate_limit_per_minute,
            'enable_prometheus': self.monitoring.enable_prometheus,
            'slow_request_threshold': self.monitoring.slow_request_threshold
        }
    
    def setup_logging(self):
        """Configure logging based on configuration."""
        # Configure root logger
        log_level = getattr(logging, self.logging.level)
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Configure handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler if specified
        if self.logging.file_path:
            from logging.handlers import RotatingFileHandler
            
            # Ensure log directory exists
            log_dir = Path(self.logging.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            format=self.logging.format
        )
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get information about the current environment."""
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "deployment_id": os.getenv("DEPLOYMENT_ID", "unknown"),
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "commit_sha": os.getenv("COMMIT_SHA", "unknown"),
            "build_timestamp": os.getenv("BUILD_TIMESTAMP", "unknown")
        }


def load_config() -> FraudDetectionConfig:
    """Load configuration singleton."""
    return FraudDetectionConfig()


def create_sample_env_file(file_path: str = ".env.example"):
    """Create a sample environment file with all configuration options."""
    sample_env = """# Fraud Detection API Configuration

# Environment
ENVIRONMENT=development
APP_VERSION=1.0.0
DEPLOYMENT_ID=local-dev

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_MAX_REQUEST_SIZE=16777216
API_ENABLE_DOCS=true
API_KEY_REQUIRED=false

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false
REDIS_TIMEOUT=5.0
REDIS_MAX_CONNECTIONS=50

# Model Configuration
MODEL_PATH=ml-models/exported
ENABLE_SHAP=false
MODEL_INFERENCE_TIMEOUT=10.0
MODEL_MAX_BATCH_SIZE=100
MODEL_FALLBACK_MODE=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE_PATH=logs/fraud_detection.log
LOG_MAX_FILE_SIZE=10485760
LOG_BACKUP_COUNT=5

# Security Configuration
API_KEYS=
RATE_LIMIT_PER_MINUTE=1000
ENABLE_REQUEST_VALIDATION=true
SECURE_HEADERS=true
TRUSTED_PROXIES=

# Monitoring Configuration
ENABLE_PROMETHEUS=true
METRICS_PATH=/metrics
HEALTH_CHECK_PATH=/health
SLOW_REQUEST_THRESHOLD=0.1
ENABLE_DETAILED_LOGGING=false
"""
    
    with open(file_path, 'w') as f:
        f.write(sample_env)
    
    print(f"Sample environment file created: {file_path}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"API will run on {config.api.host}:{config.api.port}")
    print(f"Redis connection: {config.redis.host}:{config.redis.port}")
    print(f"Model path: {config.model.model_path}")
    
    # Create sample environment file
    create_sample_env_file()
