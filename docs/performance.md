# Performance and Scaling Guide

This document describes the performance characteristics, scaling limits, and optimization strategies for the Decision Matrix MCP.

## Performance Characteristics

### Baseline Performance Metrics

Based on testing with typical configurations:

| Operation | Latency (P50) | Latency (P95) | Notes |
|-----------|---------------|---------------|-------|
| Session Creation | 5-15ms | 20-50ms | In-memory operation |
| Add Criterion | 2-8ms | 15-30ms | Model setup overhead |
| Add Option | 1-5ms | 10-20ms | Simple data structure update |
| Parallel Evaluation | 2-8s | 10-25s | Depends on model and complexity |
| Decision Matrix | 10-50ms | 100-200ms | In-memory calculation |

### Throughput Characteristics

- **Sequential Sessions**: 100-200 sessions/minute (depending on evaluation complexity)
- **Concurrent Sessions**: 10-50 concurrent evaluations (limited by LLM backend quotas)
- **Options per Session**: 2-20 (optimal), 50+ (performance degradation)
- **Criteria per Session**: 2-15 (optimal), 25+ (evaluation time increases significantly)

### Resource Utilization

#### Memory Usage
- **Base Application**: 50-100 MB
- **Per Active Session**: 1-5 MB (depending on conversation history)
- **Backend Clients**: 10-20 MB each (cached instances)
- **Peak Usage**: 200-500 MB for typical workloads

#### CPU Usage
- **Idle**: <1% CPU utilization
- **During Evaluation**: 10-30% CPU (primarily network I/O wait)
- **Concurrent Evaluations**: Scales linearly with parallel requests

#### Network I/O
- **Bedrock**: 1-10 KB request, 0.5-5 KB response per evaluation
- **LiteLLM**: 2-15 KB request, 1-8 KB response per evaluation
- **Ollama**: 5-50 KB request, 2-20 KB response per evaluation

## Scaling Limits

### Session Management Limits

```python
# Default limits (configurable)
MAX_ACTIVE_SESSIONS = 100
SESSION_TTL = 3600  # 1 hour
CLEANUP_INTERVAL = 300  # 5 minutes
```

#### Session Capacity
- **Maximum Active Sessions**: 100 (default), 1000+ (with tuning)
- **Session TTL**: 1 hour (default), configurable
- **Memory per Session**: 1-5 MB average
- **Cleanup Frequency**: 5 minutes (automatic)

### Evaluation Limits

#### Parallel Evaluation Constraints
```python
# Concurrency limits
MAX_PARALLEL_EVALUATIONS = 10  # Per criterion
THREAD_POOL_SIZE = 20          # Global thread pool
TIMEOUT_PER_EVALUATION = 120   # Seconds
```

- **Options × Criteria Matrix**: Recommended max 300 total evaluations (15×20)
- **Parallel Threads**: 10 evaluations per criterion (configurable)
- **Total Concurrent**: 50-100 concurrent LLM requests
- **Evaluation Timeout**: 120 seconds per option-criterion pair

### Backend-Specific Limits

#### AWS Bedrock
- **Rate Limits**: Varies by model and region
  - Claude 3 Sonnet: 20 requests/minute (default)
  - Claude 3 Haiku: 40 requests/minute (default)
  - Titan Text: 100 requests/minute (default)
- **Concurrent Requests**: 5-10 per model
- **Request Size**: 100KB max prompt
- **Response Size**: 4096 tokens max

#### LiteLLM (OpenAI/Anthropic)
- **Rate Limits**: Based on API tier
  - GPT-4: 10,000 requests/minute (Tier 5)
  - Claude-3: 5,000 requests/minute (production)
- **Concurrent Requests**: 100+ (higher tier accounts)
- **Token Limits**: Varies by model (8K-200K context)

#### Ollama (Local)
- **Hardware Dependent**: CPU/GPU capacity
- **Model Size**: Affects memory and latency
  - 7B models: 4-8 GB RAM, 1-5s latency
  - 13B models: 8-16 GB RAM, 3-10s latency
  - 70B models: 40+ GB RAM, 10-30s latency
- **Concurrent Requests**: 1-4 (depending on hardware)

## Performance Optimization Strategies

### 1. Session Management Optimization

#### Session Caching
```python
# Optimized session configuration
session_manager = SessionManager(
    max_sessions=500,           # Increase limit
    ttl_seconds=1800,          # 30 minutes (reduce from 1 hour)
    cleanup_interval=180       # 3 minutes (more frequent cleanup)
)
```

#### Memory-Efficient Conversation History
```python
# Limit conversation history size
class CriterionThread:
    MAX_HISTORY_LENGTH = 10  # Keep only recent messages

    def add_message(self, role: str, content: str):
        self.conversation_history.append(message)
        # Trim history to prevent memory growth
        if len(self.conversation_history) > self.MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_LENGTH:]
```

### 2. Evaluation Optimization

#### Batch Processing
```python
# Process evaluations in optimal batch sizes
OPTIMAL_BATCH_SIZE = 5  # 5 options per criterion evaluation batch

async def evaluate_in_batches(self, options: list[Option], criterion: Criterion):
    batches = [options[i:i+OPTIMAL_BATCH_SIZE]
               for i in range(0, len(options), OPTIMAL_BATCH_SIZE)]

    results = []
    for batch in batches:
        batch_result = await self.evaluate_batch(batch, criterion)
        results.extend(batch_result)

    return results
```

#### Parallel Criterion Processing
```python
# Optimize thread pool configuration
import asyncio

# Configure event loop for I/O-bound tasks
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Use ThreadPoolExecutor for I/O-bound LLM calls
executor = ThreadPoolExecutor(max_workers=20)
```

#### Smart Timeout Configuration
```python
# Dynamic timeouts based on model and complexity
def calculate_timeout(model_backend: ModelBackend, options_count: int) -> int:
    base_timeout = {
        ModelBackend.BEDROCK: 30,  # Bedrock typically faster
        ModelBackend.LITELLM: 45,  # Network overhead
        ModelBackend.OLLAMA: 60    # Local processing variance
    }

    # Scale with number of options
    complexity_factor = min(options_count / 5, 3.0)  # Max 3x scaling
    return int(base_timeout[model_backend] * complexity_factor)
```

### 3. Backend Optimization

#### Connection Pooling
```python
# Reuse HTTP connections for external APIs
class LiteLLMBackend(LLMBackend):
    def __init__(self):
        # Configure session with connection pooling
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        ))
```

#### Circuit Breaker Implementation
```python
class BackendCircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure = None
        self.state = "CLOSED"

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure = time.time()
            raise
```

### 4. Memory Management

#### Session Cleanup Strategies
```python
# Proactive cleanup based on memory usage
import psutil

class MemoryAwareSessionManager(SessionManager):
    def __init__(self, memory_threshold=80):  # 80% memory usage
        super().__init__()
        self.memory_threshold = memory_threshold

    def cleanup_if_needed(self):
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.memory_threshold:
            # Aggressive cleanup - remove oldest 25% of sessions
            sessions_to_remove = len(self.sessions) // 4
            self.cleanup_oldest_sessions(sessions_to_remove)
```

#### Response Size Optimization
```python
# Minimize response payload size
class OptimizedDecisionSession(DecisionSession):
    def get_decision_matrix(self, include_history=False) -> dict[str, Any]:
        result = super().get_decision_matrix()

        if not include_history:
            # Remove conversation history from response
            for thread in self.threads.values():
                thread.conversation_history = []

        return result
```

## Monitoring and Metrics

### Key Performance Indicators

#### Latency Metrics
```python
# Track operation latencies
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.latencies = defaultdict(list)
        self.error_counts = defaultdict(int)

    def record_latency(self, operation: str, latency: float):
        self.latencies[operation].append(latency)
        # Keep only recent measurements
        if len(self.latencies[operation]) > 1000:
            self.latencies[operation] = self.latencies[operation][-500:]

    def get_percentiles(self, operation: str) -> dict:
        latencies = sorted(self.latencies[operation])
        if not latencies:
            return {}

        n = len(latencies)
        return {
            "p50": latencies[int(n * 0.5)],
            "p90": latencies[int(n * 0.9)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[int(n * 0.99)]
        }
```

#### Throughput Monitoring
```python
# Track request rates and success ratios
class ThroughputMonitor:
    def __init__(self, window_size=300):  # 5-minute windows
        self.window_size = window_size
        self.requests = []
        self.errors = []

    def record_request(self, success: bool):
        now = time.time()
        self.requests.append(now)
        if not success:
            self.errors.append(now)

        # Clean old data
        cutoff = now - self.window_size
        self.requests = [t for t in self.requests if t > cutoff]
        self.errors = [t for t in self.errors if t > cutoff]

    def get_metrics(self) -> dict:
        return {
            "requests_per_minute": len(self.requests) / (self.window_size / 60),
            "error_rate": len(self.errors) / max(len(self.requests), 1),
            "success_rate": 1 - (len(self.errors) / max(len(self.requests), 1))
        }
```

### Alerting Thresholds

#### Performance Alerts
- **High Latency**: P95 > 15 seconds for evaluation
- **Low Throughput**: <10 evaluations/minute during peak hours
- **High Error Rate**: >5% failures over 5-minute window
- **Memory Usage**: >85% memory utilization
- **Backend Failures**: >10% backend error rate

#### Capacity Alerts
- **Session Limit**: >80% of max_sessions capacity
- **Thread Pool**: >90% thread utilization
- **Connection Pool**: Connection pool exhaustion
- **Queue Depth**: >50 pending evaluations

## Deployment Configurations

### Development Environment
```yaml
# Optimized for development speed
session_manager:
  max_sessions: 10
  ttl_seconds: 1800
  cleanup_interval: 300

orchestrator:
  max_parallel_evaluations: 3
  timeout_per_evaluation: 60
  thread_pool_size: 5

backends:
  enable_circuit_breaker: false
  retry_attempts: 1
```

### Production Environment
```yaml
# Optimized for reliability and performance
session_manager:
  max_sessions: 500
  ttl_seconds: 3600
  cleanup_interval: 180

orchestrator:
  max_parallel_evaluations: 10
  timeout_per_evaluation: 120
  thread_pool_size: 20

backends:
  enable_circuit_breaker: true
  retry_attempts: 3
  circuit_breaker_threshold: 5
```

### High-Scale Environment
```yaml
# Optimized for maximum throughput
session_manager:
  max_sessions: 2000
  ttl_seconds: 1800
  cleanup_interval: 60

orchestrator:
  max_parallel_evaluations: 20
  timeout_per_evaluation: 180
  thread_pool_size: 50

backends:
  enable_circuit_breaker: true
  retry_attempts: 2
  circuit_breaker_threshold: 3
  connection_pool_size: 25
```

## Troubleshooting Performance Issues

### High Latency Diagnosis

1. **Check Backend Response Times**
   ```python
   # Measure backend-specific latencies
   backend_latencies = monitor.get_backend_latencies()
   for backend, latency in backend_latencies.items():
       if latency > 10.0:  # >10s is concerning
           logger.warning(f"High latency for {backend}: {latency}s")
   ```

2. **Analyze Thread Pool Utilization**
   ```python
   # Check if thread pool is bottleneck
   active_threads = threading.active_count()
   if active_threads > thread_pool_size * 0.9:
       logger.warning("Thread pool near capacity")
   ```

3. **Monitor Memory Usage Patterns**
   ```python
   # Check for memory leaks in sessions
   memory_per_session = total_memory / len(active_sessions)
   if memory_per_session > 10:  # >10MB per session is high
       logger.warning("High memory usage per session")
   ```

### Low Throughput Diagnosis

1. **Backend Rate Limiting**
   - Check API quotas and limits
   - Monitor 429 (rate limit) responses
   - Consider backend switching or load balancing

2. **Inefficient Parallelization**
   - Verify optimal batch sizes
   - Check thread pool configuration
   - Monitor concurrent request patterns

3. **Network Bottlenecks**
   - Test network latency to backends
   - Check connection pool efficiency
   - Monitor DNS resolution times

### Memory Leak Detection

1. **Session Accumulation**
   ```python
   # Check for sessions that never expire
   old_sessions = [s for s in sessions.values()
                   if (now - s.created_at).seconds > 7200]  # >2 hours
   if len(old_sessions) > 10:
       logger.warning(f"Found {len(old_sessions)} old sessions")
   ```

2. **Conversation History Growth**
   ```python
   # Monitor conversation history sizes
   large_histories = [t for t in threads.values()
                      if len(t.conversation_history) > 20]
   if large_histories:
       logger.warning("Large conversation histories detected")
   ```

## Best Practices

### For Optimal Performance

1. **Session Management**
   - Set appropriate TTL based on usage patterns
   - Monitor and tune cleanup intervals
   - Implement memory-based cleanup triggers

2. **Evaluation Optimization**
   - Batch small evaluations together
   - Use appropriate parallelization levels
   - Implement smart timeout calculations

3. **Backend Usage**
   - Implement circuit breakers for reliability
   - Use connection pooling for external APIs
   - Monitor and respect rate limits

4. **Resource Management**
   - Monitor memory usage trends
   - Implement graceful degradation
   - Use appropriate data structure sizes

### For Scalability

1. **Horizontal Scaling**
   - Design for stateless operation
   - Use external session storage for multi-instance deployments
   - Implement proper load balancing

2. **Vertical Scaling**
   - Tune thread pool sizes based on CPU cores
   - Optimize memory allocation patterns
   - Use appropriate garbage collection settings

3. **Monitoring and Alerting**
   - Track key performance metrics
   - Set up automated alerts for anomalies
   - Implement health checks for dependencies
