# Configuration Management

The Synapse AI Fraud Detection system uses environment-based configuration for production deployment. This replaces hardcoded values with flexible, environment-specific settings.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your specific values:
   ```bash
   # Required settings
   REDIS_HOST=your-redis-host
   REDIS_PORT=6379
   MODEL_PATH=path/to/your/models
   
   # Optional settings
   API_PORT=8000
   LOG_LEVEL=INFO
   ```

3. Run the application:
   ```bash
   python -m api-service.main
   ```

## Configuration Categories

### API Configuration
- `API_HOST`: Server bind address (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `API_WORKERS`: Number of worker processes (default: 1)
- `API_ENABLE_DOCS`: Enable Swagger docs (default: true)
- `API_KEY_REQUIRED`: Require API key authentication (default: false)
- `CORS_ORIGINS`: Allowed CORS origins, comma-separated

### Redis Configuration
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis password (optional)
- `REDIS_SSL`: Enable SSL connection (default: false)
- `REDIS_TIMEOUT`: Connection timeout in seconds (default: 5.0)
- `REDIS_MAX_CONNECTIONS`: Maximum connection pool size (default: 50)

### Model Configuration
- `MODEL_PATH`: Path to trained model files (default: ml-models/exported)
- `ENABLE_SHAP`: Enable SHAP explanations (default: false)
- `MODEL_INFERENCE_TIMEOUT`: Model timeout in seconds (default: 10.0)
- `MODEL_FALLBACK_MODE`: Enable rule-based fallback (default: true)

### Logging Configuration
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `LOG_FORMAT`: Log message format
- `LOG_FILE_PATH`: Log file path (optional, logs to console if not set)
- `LOG_MAX_FILE_SIZE`: Maximum log file size in bytes (default: 10MB)
- `LOG_BACKUP_COUNT`: Number of backup log files (default: 5)

### Security Configuration
- `API_KEYS`: Valid API keys, comma-separated (if API_KEY_REQUIRED=true)
- `RATE_LIMIT_PER_MINUTE`: Request rate limit per minute (default: 1000)
- `ENABLE_REQUEST_VALIDATION`: Enable input validation (default: true)
- `SECURE_HEADERS`: Add security headers to responses (default: true)
- `TRUSTED_PROXIES`: Trusted proxy IPs, comma-separated

### Monitoring Configuration
- `ENABLE_PROMETHEUS`: Enable Prometheus metrics (default: true)
- `METRICS_PATH`: Metrics endpoint path (default: /metrics)
- `HEALTH_CHECK_PATH`: Health check endpoint path (default: /health)
- `SLOW_REQUEST_THRESHOLD`: Slow request threshold in seconds (default: 0.1)
- `ENABLE_DETAILED_LOGGING`: Enable detailed request logging (default: false)

## Environment-Specific Configurations

### Development
Use `.env.example` as a starting point. Enable docs and detailed logging:
```
ENVIRONMENT=development
API_ENABLE_DOCS=true
LOG_LEVEL=DEBUG
ENABLE_DETAILED_LOGGING=true
```

### Production
Use `.env.production` as a template. Key differences:
- Disable API docs for security
- Require API key authentication
- Restrict CORS origins
- Use WARNING log level
- Enable SSL for Redis
- Lower rate limits

```
ENVIRONMENT=production
API_ENABLE_DOCS=false
API_KEY_REQUIRED=true
LOG_LEVEL=WARNING
REDIS_SSL=true
CORS_ORIGINS=https://yourdomain.com
```

### Docker/Kubernetes
For containerized deployments:
1. Use environment variables in your deployment manifests
2. Store secrets in Kubernetes secrets or Docker secrets
3. Use external Redis service
4. Mount model files as volumes

## Validation

The configuration system validates all settings on startup:
- Port numbers must be valid (1-65535)
- Required fields cannot be empty
- Log levels must be valid
- File paths are checked for existence where applicable

Invalid configuration will cause the application to exit with an error message.

## Migration from Legacy Configuration

If you have existing hardcoded configurations:

1. **Identify Current Values**: Note your current Redis host, model path, etc.
2. **Create Environment File**: Copy `.env.example` and update with your values
3. **Test Configuration**: Run with new config and verify all services connect
4. **Remove Legacy Code**: The new system replaces all hardcoded config dictionaries

## Best Practices

1. **Never commit `.env` files**: Add `.env` to `.gitignore`
2. **Use different configs per environment**: `.env.dev`, `.env.staging`, `.env.prod`
3. **Validate in CI/CD**: Test configuration loading in your deployment pipeline
4. **Monitor configuration**: Log startup config (excluding secrets) for debugging
5. **Use secrets management**: For production, use proper secret storage systems

## Troubleshooting

### Common Issues

1. **"Configuration validation errors"**
   - Check required fields are set
   - Verify port numbers are valid integers
   - Ensure log level is uppercase (INFO, not info)

2. **"Redis connection failed"**
   - Verify REDIS_HOST and REDIS_PORT
   - Check if Redis password is required
   - Ensure Redis is running and accessible

3. **"Model service not initialized"**
   - Check MODEL_PATH exists and contains model files
   - Verify file permissions
   - Enable fallback mode for graceful degradation

4. **"CORS errors in browser"**
   - Add your frontend URL to CORS_ORIGINS
   - Use comma-separated list for multiple origins
   - Check for typos in URLs

### Debug Mode

Enable debug logging to see configuration loading:
```
LOG_LEVEL=DEBUG
ENABLE_DETAILED_LOGGING=true
```

This will log all configuration values (except secrets) on startup.
