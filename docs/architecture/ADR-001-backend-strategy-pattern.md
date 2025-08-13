# ADR-001: Backend Strategy Pattern for LLM Integration

## Status
Accepted

## Context

The Decision Matrix MCP needs to support multiple Large Language Model (LLM) providers to ensure flexibility, avoid vendor lock-in, and provide fallback options when services are unavailable. The system needs to support:

- **AWS Bedrock** - Enterprise-grade models with Claude, Titan, and others
- **LiteLLM** - Unified interface to OpenAI, Anthropic, and other providers
- **Ollama** - Local open-source models for privacy and cost control

Key requirements:
- Each criterion should be able to use a different LLM backend for specialized evaluation
- Backend switching should be transparent to the orchestration layer
- The system should gracefully handle unavailable backends
- New backends should be easy to add without modifying existing code
- Backend instances should be reusable to avoid initialization overhead

## Decision

We implement the **Strategy Pattern** combined with a **Factory Pattern** and **Singleton instances** for LLM backend management:

1. **Strategy Pattern**: `LLMBackend` abstract base class defines the interface that all backend implementations must follow
2. **Factory Pattern**: `BackendFactory` creates and manages backend instances based on `ModelBackend` enum values
3. **Singleton Instances**: Each backend type is instantiated once and reused across evaluations

### Key Design Elements

- **Abstract Base Class** (`LLMBackend`): Defines `generate_response()` and `is_available()` methods
- **Concrete Implementations**: `BedrockBackend`, `LiteLLMBackend`, `OllamaBackend`
- **Factory Class** (`BackendFactory`): Handles creation, caching, and lifecycle management
- **Enum-Driven Selection**: `ModelBackend` enum provides type-safe backend identification

## Consequences

### Positive
- **Flexible Provider Selection**: Each criterion can specify its preferred LLM backend
- **Easy Extension**: New backends only require implementing the `LLMBackend` interface
- **Performance**: Singleton pattern avoids repeated initialization of expensive backend connections
- **Reliability**: Graceful degradation when backends are unavailable
- **Testability**: Strategy pattern enables easy mocking of backend behavior
- **Separation of Concerns**: Backend-specific logic is isolated from orchestration

### Negative
- **Increased Complexity**: More classes and indirection compared to direct API calls
- **Memory Overhead**: Singleton instances remain in memory for application lifetime
- **Potential Shared State**: Backend instances are shared across threads (mitigated by stateless design)

### Neutral
- **Configuration Complexity**: Each backend requires its own configuration (environment variables, credentials)
- **Dependency Management**: Each backend brings its own set of Python dependencies

## Alternatives Considered

### Alternative 1: Direct Backend Instantiation
**Description**: Create backend instances directly in the orchestrator without factory pattern.
**Pros**: Simpler, fewer classes, direct control over instances
**Cons**: Tight coupling, harder to test, no reuse of expensive connections
**Reason for rejection**: Violates single responsibility principle and makes testing difficult

### Alternative 2: Service Locator Pattern
**Description**: Central registry where backends register themselves and are looked up by string names.
**Pros**: Very flexible, runtime backend registration
**Cons**: Runtime errors for missing backends, harder to track dependencies
**Reason for rejection**: Less type safety and harder to reason about availability

### Alternative 3: Dependency Injection Framework
**Description**: Use external DI framework (e.g., `dependency-injector`) for backend management.
**Pros**: Industry standard approach, sophisticated lifecycle management
**Cons**: Additional dependency, overkill for this specific use case
**Reason for rejection**: The simple factory pattern meets all requirements without external dependencies

## Implementation Notes

### Key Files
- `src/decision_matrix_mcp/backends/base.py` - Abstract base class definition
- `src/decision_matrix_mcp/backends/factory.py` - Factory implementation with singleton pattern
- `src/decision_matrix_mcp/backends/bedrock.py` - AWS Bedrock implementation
- `src/decision_matrix_mcp/backends/litellm.py` - LiteLLM proxy implementation
- `src/decision_matrix_mcp/backends/ollama.py` - Local Ollama implementation

### Configuration
Each backend checks for its own dependencies and configuration:
- **Bedrock**: Requires `boto3` and AWS credentials
- **LiteLLM**: Requires `litellm` package and API keys in environment
- **Ollama**: Requires `requests` and local Ollama server running

### Thread Safety
- Backend instances are stateless and thread-safe
- Factory maintains thread-safe singleton cache
- No shared mutable state between concurrent evaluations

### Error Handling
- `is_available()` method allows graceful degradation
- Specific exceptions for different failure modes:
  - `LLMConfigurationError` for missing dependencies
  - `LLMAPIError` for network/API failures
  - `LLMBackendError` for backend-specific issues

## References
- [Strategy Pattern - Gang of Four](https://en.wikipedia.org/wiki/Strategy_pattern)
- [Factory Pattern documentation](https://refactoring.guru/design-patterns/factory-method)
- [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/)
- [LiteLLM documentation](https://docs.litellm.ai/)
- [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

---
**Decision Date:** 2025-08-12
**Last Updated:** 2025-08-12
**Author(s):** Claude Code
**Review Status:** Approved
