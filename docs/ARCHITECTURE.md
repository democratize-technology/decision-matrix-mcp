# Decision Matrix MCP Architecture

## Overview

Decision Matrix MCP is a Model Context Protocol server that implements structured decision analysis through parallel criterion evaluation. The architecture is designed for scalability, maintainability, and clean separation of concerns.

## System Architecture

```
┌─────────────────────┐
│   MCP Client        │
│ (Claude Desktop)    │
└──────────┬──────────┘
           │ stdio (JSON-RPC)
           │
┌──────────▼──────────┐
│   MCP Server        │
│  (FastMCP Framework)│
└──────────┬──────────┘
           │
┌──────────▼──────────┐     ┌─────────────────┐
│  Request Handlers   │────▶│  Validation     │
│  (Tool Functions)   │     │  Decorators     │
└──────────┬──────────┘     └─────────────────┘
           │
┌──────────▼──────────┐     ┌─────────────────┐
│  Session Manager    │────▶│  Decision       │
│  (State Management) │     │  Sessions       │
└──────────┬──────────┘     └─────────────────┘
           │
┌──────────▼──────────┐     ┌─────────────────┐
│  Orchestrator       │────▶│  LLM Backends   │
│  (Parallel Eval)    │     │  (Multi-vendor) │
└─────────────────────┘     └─────────────────┘
```

## Core Components

### 1. MCP Server Layer (`__init__.py`)

The entry point and MCP protocol implementation:

- **FastMCP Integration**: Uses FastMCP for simplified MCP server creation
- **Tool Registration**: Exposes decision analysis tools to MCP clients
- **Dependency Injection**: ServerComponents container for lifecycle management
- **Error Handling**: Consistent error responses across all tools

Key features:
- Stdio transport (stdout for protocol, stderr for logging)
- Request/response validation with Pydantic
- Session-based state management
- Graceful connection handling

### 2. Session Management (`session_manager.py`)

Manages decision analysis sessions with proper isolation:

```python
SessionManager
├── create_session(topic, options)
├── get_session(session_id)
├── remove_session(session_id)
├── list_active_sessions()
└── _cleanup_expired_sessions()
```

Features:
- UUID-based session identification
- TTL-based expiration (24 hours default)
- Resource limits (max concurrent sessions)
- Automatic cleanup on interval
- Thread-safe operations

### 3. Orchestrator (`orchestrator.py`)

Handles parallel evaluation across criteria:

```python
DecisionOrchestrator
├── evaluate_options_across_criteria(threads, options)
├── _evaluate_single_option(thread, option)
├── _get_llm_response(thread, option, prompt)
└── _parse_evaluation_response(response)
```

Key design decisions:
- **Parallel Execution**: Uses asyncio.gather for concurrent evaluations
- **Thread Isolation**: Each criterion maintains conversation context
- **Multi-backend Support**: Abstracted LLM interface
- **Retry Logic**: Exponential backoff for transient failures
- **Graceful Degradation**: Handle abstentions and errors

### 4. Data Models (`models.py`)

Type-safe data structures using Pydantic:

```python
DecisionSession
├── Options (dict[str, Option])
│   └── Scores (dict[str, Score])
├── Criteria (dict[str, Criterion])
│   └── CriterionThreads
└── Evaluations (list[dict])
```

Model relationships:
- **Session**: Root aggregate containing all decision data
- **Option**: Choice being evaluated with score collection
- **Criterion**: Evaluation dimension with weight and prompt
- **Score**: Individual evaluation result (nullable for abstentions)
- **CriterionThread**: Conversation context per criterion

### 5. LLM Backend Integration

Multi-vendor support with consistent interface:

```python
Backend Interface
├── AWS Bedrock (boto3)
│   └── Claude models
├── LiteLLM (HTTP)
│   ├── OpenAI
│   └── Anthropic
└── Ollama (HTTP)
    └── Local models
```

Each backend implements:
- Authentication handling
- Request formatting
- Response parsing
- Error mapping
- Retry logic

### 6. Validation Layer (`validation_decorators.py`)

Centralized input validation:

```python
@validate_request(
    topic=SessionValidator.validate_topic,
    options=SessionValidator.validate_option_name
)
async def start_decision_analysis(request):
    # Implementation with pre-validated inputs
```

Benefits:
- Consistent validation across endpoints
- Clear error messages
- DRY principle
- Type safety

## Design Patterns

### 1. Dependency Injection

```python
# No global state
components = ServerComponents(
    orchestrator=DecisionOrchestrator(),
    session_manager=SessionManager()
)

# Injected per request
def get_server_components() -> ServerComponents:
    return _server_components
```

### 2. Repository Pattern

Session Manager acts as repository:
- Encapsulates storage logic
- Provides domain-specific queries
- Handles persistence lifecycle

### 3. Strategy Pattern

LLM backends as strategies:
- Common interface
- Runtime selection
- Easy to add new backends

### 4. Decorator Pattern

Validation decorators:
- Wrap handler functions
- Add validation behavior
- Preserve function signatures

## Concurrency Model

### Thread Safety

- **Session Isolation**: Each session is independent
- **Immutable Messages**: Request/response objects are immutable
- **Async Operations**: Non-blocking I/O for LLM calls

### Parallel Evaluation

```python
# All criterion-option pairs evaluated concurrently
tasks = [
    evaluate(criterion1, option1),
    evaluate(criterion1, option2),
    evaluate(criterion2, option1),
    evaluate(criterion2, option2)
]
results = await asyncio.gather(*tasks)
```

## Error Handling Strategy

### Error Hierarchy

```python
DecisionMatrixError (base)
├── ValidationError (user input)
├── SessionError (state issues)
├── ResourceLimitError (quotas)
├── LLMBackendError (external)
│   ├── LLMConfigurationError
│   └── LLMAPIError
└── ConfigurationError (setup)
```

### Error Propagation

1. **Validation Layer**: Catch input errors early
2. **Business Logic**: Domain-specific exceptions
3. **Infrastructure**: Backend/transport errors
4. **Handler Layer**: Convert to user-friendly messages

## Security Considerations

### Input Validation

- All inputs validated with Pydantic
- Length limits on strings
- Range constraints on numbers
- UUID format validation

### Prompt Injection Prevention

- System prompts not user-modifiable
- Custom prompts are optional additions
- Output parsing is defensive

### Resource Protection

- Max sessions per instance
- Evaluation timeouts
- Memory limits via Docker

## Performance Characteristics

### Scalability

- **Horizontal**: Multiple MCP server instances
- **Vertical**: Concurrent evaluation threads
- **Caching**: Not implemented (stateless evaluations)

### Bottlenecks

1. **LLM Latency**: Primary constraint (1-5s per evaluation)
2. **Memory**: Session storage (minimal per session)
3. **CPU**: Negligible (I/O bound workload)

### Optimization Strategies

- Parallel evaluation across all pairs
- Connection pooling for HTTP backends
- Exponential backoff for retries
- Early termination on critical errors

## Extension Points

### Adding New LLM Backends

1. Implement backend method in Orchestrator
2. Add to ModelBackend enum
3. Handle authentication
4. Map errors appropriately

### Custom Criteria Types

1. Extend Criterion model
2. Add specialized prompts
3. Implement custom scoring logic

### Alternative Scoring Methods

1. Replace Score model
2. Update matrix generation
3. Modify ranking algorithm

## Deployment Architecture

### Container Deployment

```
Docker Container
├── Python Runtime
├── Application Code
├── Dependencies
└── Non-root User
```

### Environment Configuration

```bash
# LLM Backend Selection
AWS_PROFILE=default
LITELLM_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434

# Operational
LOG_LEVEL=WARNING
SESSION_TTL_HOURS=24
MAX_SESSIONS=100
```

### Monitoring Points

- Session creation/expiration
- Evaluation success/failure rates
- LLM backend latency
- Error frequencies by type

## Future Considerations

### Potential Enhancements

1. **Persistent Storage**: Database backing for sessions
2. **Webhooks**: Notification on completion
3. **Batch Operations**: Multiple decisions in parallel
4. **Templates**: Pre-defined decision frameworks
5. **Analytics**: Decision pattern analysis

### Scaling Strategies

1. **Session Sharding**: Distribute by session ID
2. **Read Replicas**: For session queries
3. **Queue-based**: Async evaluation jobs
4. **Caching Layer**: For repeated evaluations

This architecture provides a solid foundation for structured decision analysis while maintaining the flexibility to evolve based on user needs and scale requirements.
