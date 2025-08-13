# Configuration Management

The Decision Matrix MCP now supports comprehensive configuration via environment variables, removing hardcoded limits and enabling flexible deployment across different environments.

## Quick Start

All configuration is optional with sensible defaults. Set environment variables with the `DMM_` prefix to customize behavior:

```bash
# Basic configuration
export DMM_MAX_OPTIONS_ALLOWED=30
export DMM_MAX_ACTIVE_SESSIONS=200
export DMM_ENVIRONMENT=development

# Run the server
python3 -m decision_matrix_mcp
```

## Environment Variables

### Validation Limits

Control input validation and business logic constraints:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DMM_MAX_SESSION_ID_LENGTH` | int | 100 | Maximum length for session IDs |
| `DMM_MAX_TOPIC_LENGTH` | int | 500 | Maximum length for decision topics |
| `DMM_MAX_OPTION_NAME_LENGTH` | int | 200 | Maximum length for option names |
| `DMM_MAX_CRITERION_NAME_LENGTH` | int | 100 | Maximum length for criterion names |
| `DMM_MAX_DESCRIPTION_LENGTH` | int | 1000 | Maximum length for descriptions |
| `DMM_MIN_OPTIONS_REQUIRED` | int | 2 | Minimum number of options required |
| `DMM_MAX_OPTIONS_ALLOWED` | int | 20 | Maximum number of options allowed |
| `DMM_MAX_CRITERIA_ALLOWED` | int | 10 | Maximum number of criteria allowed |
| `DMM_MIN_CRITERION_WEIGHT` | float | 0.1 | Minimum weight value for criteria |
| `DMM_MAX_CRITERION_WEIGHT` | float | 10.0 | Maximum weight value for criteria |

### Session Management

Control session lifecycle and memory usage:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DMM_MAX_ACTIVE_SESSIONS` | int | 100 | Maximum sessions before LRU eviction |
| `DMM_LRU_EVICTION_BATCH_SIZE` | int | 10 | Sessions to evict per cleanup batch |
| `DMM_DEFAULT_MAX_SESSIONS` | int | 50 | Default max sessions per manager |
| `DMM_DEFAULT_SESSION_TTL_HOURS` | int | 24 | Default session time-to-live |
| `DMM_DEFAULT_CLEANUP_INTERVAL_MINUTES` | int | 30 | Default cleanup interval |

### Performance & Concurrency

Control timeouts, retries, and concurrent operations:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DMM_MAX_RETRIES` | int | 3 | Maximum retry attempts for evaluations |
| `DMM_RETRY_DELAY_SECONDS` | float | 1.0 | Initial delay between retries |
| `DMM_MAX_CONCURRENT_EVALUATIONS` | int | 10 | Maximum concurrent evaluations |
| `DMM_REQUEST_TIMEOUT_SECONDS` | float | 30.0 | Default timeout for LLM requests |
| `DMM_COT_TIMEOUT_SECONDS` | float | 30.0 | Timeout for Chain of Thought evaluation |
| `DMM_COT_SUMMARY_TIMEOUT_SECONDS` | float | 5.0 | Quick timeout for CoT summary |

### Backend Configuration

Control LLM backend settings and model selection:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DMM_BEDROCK_MODEL` | str | anthropic.claude-3-sonnet-20240229-v1:0 | Default AWS Bedrock model |
| `DMM_LITELLM_MODEL` | str | gpt-4o-mini | Default LiteLLM model |
| `DMM_OLLAMA_MODEL` | str | llama3.1:8b | Default Ollama model |
| `DMM_BEDROCK_TIMEOUT_SECONDS` | float | 30.0 | Timeout for Bedrock requests |
| `DMM_LITELLM_TIMEOUT_SECONDS` | float | 30.0 | Timeout for LiteLLM requests |
| `DMM_OLLAMA_TIMEOUT_SECONDS` | float | 60.0 | Timeout for Ollama requests |
| `DMM_DEFAULT_TEMPERATURE` | float | 0.1 | Default temperature for LLM requests |

### Environment & Debug

Control environment-specific behavior:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DMM_ENVIRONMENT` | str | production | Environment mode (development/staging/production) |
| `DMM_DEBUG_MODE` | bool | false | Enable debug mode with additional logging |
| `DMM_CONFIG_VERSION` | str | 1.0.0 | Configuration schema version |

## Environment Profiles

The system automatically applies environment-specific defaults:

### Development Profile (`DMM_ENVIRONMENT=development`)
```bash
DMM_DEBUG_MODE=true
DMM_MAX_OPTIONS_ALLOWED=10        # Smaller limits
DMM_MAX_ACTIVE_SESSIONS=20        # Less memory usage
DMM_REQUEST_TIMEOUT_SECONDS=60.0  # Longer timeouts for debugging
```

### Staging Profile (`DMM_ENVIRONMENT=staging`)
```bash
DMM_DEBUG_MODE=false
DMM_MAX_OPTIONS_ALLOWED=15        # Medium limits
DMM_MAX_ACTIVE_SESSIONS=50        # Moderate memory usage
DMM_REQUEST_TIMEOUT_SECONDS=45.0  # Balanced timeouts
```

### Production Profile (`DMM_ENVIRONMENT=production`) - Default
Uses all default values optimized for production workloads.

## Configuration Files

Alternative to environment variables, create configuration files:

### JSON Configuration
```json
{
  "validation": {
    "max_options_allowed": 25,
    "max_description_length": 2000
  },
  "session": {
    "max_active_sessions": 150,
    "default_session_ttl_hours": 48
  },
  "performance": {
    "max_concurrent_evaluations": 15,
    "request_timeout_seconds": 45.0
  },
  "backend": {
    "bedrock_model": "anthropic.claude-3-opus-20240229-v1:0",
    "default_temperature": 0.3
  },
  "environment": "staging",
  "debug_mode": false
}
```

### Configuration File Locations
The system searches for configuration files in this order:

1. `/etc/decision-matrix-mcp/config.json`
2. `/etc/decision-matrix-mcp/config.yaml`
3. `~/.config/decision-matrix-mcp/config.json`
4. `~/.config/decision-matrix-mcp/config.yaml`
5. `./config.json`
6. `./config.yaml`

Environment variables always override file configuration.

## Boolean Values

Boolean environment variables accept these values:
- **True**: `true`, `1`, `yes`, `on`
- **False**: `false`, `0`, `no`, `off`

## Validation & Security

The configuration system includes comprehensive validation:

### Automatic Validation
- **Type checking**: Ensures correct data types
- **Range validation**: Enforces minimum/maximum constraints
- **Cross-field validation**: Ensures consistent relationships
- **Resource validation**: Warns about excessive memory/CPU usage
- **Security validation**: Warns about potential DoS vectors

### Configuration Warnings
The system will log warnings for:
- Very large limits that could enable DoS attacks
- Debug mode enabled in production
- Resource-intensive configurations
- Inconsistent timeout relationships

### Example Validation Output
```
[WARNING] Very large session limit (5000) could consume excessive memory
[WARNING] Debug mode enabled in production environment
[INFO] Configuration recommendations: Consider reducing concurrent evaluations
```

## Migration from Hardcoded Constants

### Backward Compatibility
Existing code continues to work unchanged:
```python
from decision_matrix_mcp.constants import ValidationLimits, SessionLimits

# Still works - now reads from configuration
max_options = ValidationLimits.MAX_OPTIONS_ALLOWED
max_sessions = SessionLimits.MAX_ACTIVE_SESSIONS
```

### New Preferred Usage
For new code, use the configuration directly:
```python
from decision_matrix_mcp.config import config

max_options = config.validation.max_options_allowed
max_sessions = config.session.max_active_sessions
timeout = config.performance.request_timeout_seconds
```

## Runtime Configuration Updates

Update configuration programmatically:
```python
from decision_matrix_mcp.config import config

# Update specific values
config.update_config(
    validation__max_options_allowed=35,
    performance__request_timeout_seconds=60.0
)

# Reload from environment/files
config.reload_configuration()
```

## Configuration Monitoring

Get configuration status and validation results:
```python
from decision_matrix_mcp.config import config

# Get comprehensive summary
summary = config.get_config_summary()
print(f"Environment: {summary['environment']}")
print(f"Warnings: {summary['validation']['warning_count']}")

# Export current configuration
json_config = config.export_config("json")
yaml_config = config.export_config("yaml")

# Get help for environment variables
help_text = config.get_env_var_help()
```

## Performance Considerations

### Memory Usage
- Each session uses ~50KB
- Each concurrent evaluation uses ~10MB
- The system warns when total estimated usage exceeds 80% of available memory

### CPU Usage
- The system recommends max concurrent evaluations ≤ 2x CPU cores
- Excessive concurrency can hurt performance due to context switching

### Timeout Tuning
- **Development**: Longer timeouts (60s+) for debugging
- **Production**: Shorter timeouts (30s) for responsiveness
- **CoT timeout**: Should be longer than request timeout
- **Summary timeout**: Should be much shorter (5s) for quick responses

## Troubleshooting

### Common Issues

1. **Invalid Environment Variable**
   ```
   [WARNING] Invalid value for DMM_MAX_OPTIONS_ALLOWED='abc': invalid literal for int()
   ```
   *Solution*: Use numeric values for integer/float variables

2. **Configuration Validation Failed**
   ```
   ConfigValidationError: max_options_allowed (3) must be greater than min_options_required (5)
   ```
   *Solution*: Ensure max values are greater than min values

3. **Resource Warnings**
   ```
   [WARNING] Configuration may use ~2048MB memory (available: 1.5GB)
   ```
   *Solution*: Reduce session limits or concurrent evaluations

### Debug Configuration
Enable debug mode to see detailed configuration loading:
```bash
export DMM_DEBUG_MODE=true
export DMM_ENVIRONMENT=development
python3 -m decision_matrix_mcp
```

This will show:
- Which environment variables were loaded
- Configuration file sources
- Validation warnings and recommendations
- Applied environment profile overrides

## Examples

### High-Performance Configuration
```bash
# For powerful servers
export DMM_MAX_ACTIVE_SESSIONS=500
export DMM_MAX_CONCURRENT_EVALUATIONS=20
export DMM_MAX_OPTIONS_ALLOWED=50
export DMM_REQUEST_TIMEOUT_SECONDS=60.0
```

### Memory-Constrained Configuration
```bash
# For limited resources
export DMM_MAX_ACTIVE_SESSIONS=25
export DMM_MAX_CONCURRENT_EVALUATIONS=3
export DMM_MAX_OPTIONS_ALLOWED=10
export DMM_DEFAULT_SESSION_TTL_HOURS=6
```

### Development Configuration
```bash
# For local development
export DMM_ENVIRONMENT=development
export DMM_DEBUG_MODE=true
export DMM_REQUEST_TIMEOUT_SECONDS=120.0
export DMM_COT_TIMEOUT_SECONDS=120.0
export DMM_MAX_OPTIONS_ALLOWED=5
```

### Production Configuration
```bash
# For production deployment
export DMM_ENVIRONMENT=production
export DMM_DEBUG_MODE=false
export DMM_MAX_ACTIVE_SESSIONS=200
export DMM_REQUEST_TIMEOUT_SECONDS=30.0
export DMM_BEDROCK_MODEL=anthropic.claude-3-opus-20240229-v1:0
```
