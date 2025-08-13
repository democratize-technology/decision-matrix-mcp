# Error Handling Guide

This document describes the error handling patterns, exception hierarchy, and troubleshooting guidance for the Decision Matrix MCP.

## Exception Hierarchy

The Decision Matrix MCP uses a structured exception hierarchy to provide clear error categorization and appropriate handling strategies.

### Base Exception Classes

```python
LLMBackendError                    # Base for all LLM-related errors
├── LLMConfigurationError          # Missing dependencies, invalid config
├── LLMAPIError                    # API call failures, authentication
└── ChainOfThoughtError           # Reasoning process failures
    ├── CoTTimeoutError           # Reasoning timeout exceeded
    └── CoTParsingError           # Response parsing failures
```

### Exception Details

#### LLMBackendError
**Base class for all LLM backend exceptions.**

```python
class LLMBackendError(Exception):
    def __init__(self, backend: str, message: str, user_message: str = None, original_error: Exception = None)
```

- `backend`: Which LLM backend failed (bedrock, litellm, ollama)
- `message`: Technical error details for logging
- `user_message`: User-friendly explanation and resolution steps
- `original_error`: Wrapped underlying exception for debugging

#### LLMConfigurationError
**Configuration or dependency issues that prevent backend usage.**

Common scenarios:
- Missing required packages (boto3, litellm, requests)
- Invalid environment variables (AWS_REGION, API keys)
- Insufficient permissions (IAM roles, API access)
- Network connectivity issues

#### LLMAPIError
**API call failures during request processing.**

Common scenarios:
- Authentication failures (invalid credentials)
- Authorization issues (insufficient permissions)
- Rate limiting (quota exceeded)
- Model availability (unsupported regions)
- Request format errors (invalid parameters)

#### ChainOfThoughtError
**Failures in the reasoning and response processing pipeline.**

#### CoTTimeoutError
**Reasoning process exceeded configured timeout.**

- Default timeout: 120 seconds per evaluation
- Configurable via environment variables
- Often indicates model performance issues

#### CoTParsingError
**Response parsing failures for structured outputs.**

- JSON parsing errors in score responses
- Missing required fields (score, justification)
- Invalid score ranges (outside 1-10 scale)

## Error Handling Patterns

### 1. Graceful Degradation

The system attempts to continue operation even when some components fail:

```python
# Backend availability checking
available_backends = factory.get_available_backends()
if ModelBackend.BEDROCK not in available_backends:
    logger.warning("Bedrock unavailable, falling back to LiteLLM")
    criterion.model_backend = ModelBackend.LITELLM
```

### 2. Abstention Handling

When criteria cannot evaluate options, they abstain rather than error:

```python
# Abstention example
score = Score(
    criterion_name="performance",
    option_name="Option A",
    score=None,  # Abstained
    justification="Performance criterion not applicable to this option type"
)
```

### 3. Retry Logic with Exponential Backoff

```python
async def evaluate_with_retry(thread: CriterionThread, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return await backend.generate_response(thread)
        except LLMAPIError as e:
            if attempt == max_retries - 1:
                raise
            delay = 2 ** attempt  # Exponential backoff
            await asyncio.sleep(delay)
```

### 4. Circuit Breaker Pattern

```python
class BackendCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.last_failure_time = None
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
```

## Common Error Scenarios

### AWS Bedrock Issues

#### Authentication Failures
**Error**: "Unable to locate credentials"
**Cause**: Missing or invalid AWS credentials
**Resolution**:
1. Configure AWS CLI: `aws configure`
2. Set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
3. Use IAM roles for EC2/ECS deployments
4. Verify with: `aws sts get-caller-identity`

#### Model Access Denied
**Error**: "ValidationException: The provided model identifier is invalid"
**Cause**: Model not available in current region or account
**Resolution**:
1. Check model availability: `aws bedrock list-foundation-models`
2. Verify region supports desired model
3. Request model access through AWS Console
4. Use alternative model ID

#### Rate Limiting
**Error**: "ThrottlingException: Rate exceeded"
**Cause**: Too many concurrent requests to Bedrock
**Resolution**:
1. Implement exponential backoff
2. Reduce concurrent evaluation threads
3. Request quota increase from AWS
4. Consider model switching for lower-traffic models

### LiteLLM Issues

#### API Key Missing
**Error**: "AuthenticationError: No API key provided"
**Cause**: Missing API key environment variable
**Resolution**:
1. Set `OPENAI_API_KEY` for OpenAI models
2. Set `ANTHROPIC_API_KEY` for Claude models
3. Use `LITELLM_API_KEY` for unified configuration
4. Verify with: `echo $OPENAI_API_KEY`

#### Model Not Found
**Error**: "NotFoundError: Model not found"
**Cause**: Invalid model name or insufficient permissions
**Resolution**:
1. Check available models in provider dashboard
2. Verify model name spelling (e.g., `gpt-4` vs `gpt-4-turbo`)
3. Ensure account has access to specified model
4. Update to supported model identifier

### Ollama Issues

#### Server Not Running
**Error**: "ConnectionError: Failed to connect to Ollama server"
**Cause**: Ollama service not started or wrong host/port
**Resolution**:
1. Start Ollama: `ollama serve`
2. Check service status: `curl http://localhost:11434/api/version`
3. Verify `OLLAMA_HOST` environment variable
4. Test model availability: `ollama list`

#### Model Not Downloaded
**Error**: "Model not found: llama2"
**Cause**: Requested model not available locally
**Resolution**:
1. Download model: `ollama pull llama2`
2. List available models: `ollama list`
3. Use alternative model name
4. Check disk space for model storage

### Session Management Issues

#### Session Not Found
**Error**: "Session abc-123 not found or expired"
**Cause**: Session expired due to TTL or was manually removed
**Resolution**:
1. Create new session with `start_decision_analysis`
2. Check active sessions with `list_sessions`
3. Verify session ID format (UUID)
4. Consider increasing session TTL

#### Validation Failures
**Error**: "No options to evaluate. Add options first."
**Cause**: Attempting evaluation without required setup
**Resolution**:
1. Add options with specified names and descriptions
2. Add criteria with weights and descriptions
3. Verify prerequisites with session status
4. Follow proper workflow order

## Debugging Strategies

### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("decision_matrix_mcp")
```

### 2. Check Backend Availability

```python
from decision_matrix_mcp.backends import BackendFactory

factory = BackendFactory()
available = factory.get_available_backends()
print(f"Available backends: {available}")
```

### 3. Test Individual Components

```python
# Test Bedrock connectivity
backend = BedrockBackend()
if backend.is_available():
    test_result = await backend.test_connection()
    print(test_result)
```

### 4. Validate Configuration

```python
# Check environment variables
import os
required_vars = ["AWS_REGION", "AWS_PROFILE"]
for var in required_vars:
    value = os.environ.get(var)
    print(f"{var}: {value or 'NOT SET'}")
```

## Error Response Formats

### MCP Tool Error Response

```json
{
  "error": "Session abc-123 not found or expired",
  "context": "session_validation",
  "suggestion": "Create a new session with start_decision_analysis",
  "formatted_output": "❌ **Error**: Session not found\n\n**Next Steps**:\n1. Create new session\n2. Verify session ID\n3. Check session timeout"
}
```

### Backend Error Response

```json
{
  "status": "error",
  "backend": "bedrock",
  "error_type": "LLMConfigurationError",
  "message": "boto3 is not installed. Please install with: pip install boto3",
  "user_message": "AWS Bedrock requires boto3. Install: pip install boto3",
  "suggestion": "Run: pip install boto3",
  "recovery_actions": [
    "Install boto3 package",
    "Configure AWS credentials",
    "Set AWS_REGION environment variable"
  ]
}
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Error Rates**
   - Backend failure percentage
   - Timeout occurrence rate
   - Authentication failure rate

2. **Performance Metrics**
   - Response latency percentiles
   - Evaluation completion rate
   - Concurrent request handling

3. **Resource Utilization**
   - Memory usage during evaluations
   - Thread pool utilization
   - Session cache hit rate

### Alerting Thresholds

- **Critical**: >10% backend failure rate
- **Warning**: >5% timeout rate
- **Info**: >1000ms average response time

### Log Analysis Patterns

```bash
# Find authentication errors
grep "AuthenticationError" application.log

# Check timeout patterns
grep "CoTTimeoutError" application.log | wc -l

# Monitor backend health
grep "Backend.*not available" application.log
```

## Recovery Procedures

### 1. Backend Failure Recovery

1. Check backend availability: `factory.validate_backend_availability(backend)`
2. Switch to alternative backend if available
3. Restart backend services if local (Ollama)
4. Verify credentials and permissions
5. Check service quotas and limits

### 2. Session Corruption Recovery

1. Remove corrupted session: `session_manager.remove_session(session_id)`
2. Clear session cache if persistent storage used
3. Recreate session with same configuration
4. Verify data integrity before proceeding

### 3. Memory Leak Recovery

1. Monitor memory usage trends
2. Implement session cleanup routines
3. Restart service if memory usage exceeds thresholds
4. Review session TTL configuration
5. Check for unclosed backend connections

## Best Practices

### For Developers

1. **Always use specific exception types** rather than generic Exception
2. **Include user-friendly messages** in all custom exceptions
3. **Log errors with appropriate levels** (ERROR for failures, WARNING for degradation)
4. **Provide recovery suggestions** in error messages
5. **Test error paths** as thoroughly as success paths

### For Operators

1. **Monitor error rates and patterns** for early problem detection
2. **Set up automated alerting** for critical error thresholds
3. **Maintain runbooks** for common failure scenarios
4. **Test backup procedures** regularly
5. **Keep error logs** for post-incident analysis

### For Users

1. **Check backend availability** before starting complex analyses
2. **Use appropriate timeouts** for long-running evaluations
3. **Handle abstentions gracefully** in result processing
4. **Verify prerequisites** before evaluation requests
5. **Save session IDs** for result retrieval
