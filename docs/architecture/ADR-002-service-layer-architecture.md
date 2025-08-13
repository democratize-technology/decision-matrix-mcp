# ADR-002: Service Layer Architecture Extraction

## Status
Accepted

## Context

The original Decision Matrix MCP implementation had all business logic embedded within the main MCP server handlers (`__init__.py`). This created several challenges:

- **Monolithic Structure**: Single file contained session management, validation, orchestration, and response formatting
- **Testing Difficulties**: Business logic tightly coupled to MCP protocol handling
- **Code Reusability**: Core decision analysis logic couldn't be used outside MCP context
- **Maintenance Burden**: Changes to business logic required modifying protocol-level code
- **Single Responsibility Violations**: Handlers responsible for both protocol and business concerns

The system needed clear separation between:
1. **Protocol Layer**: MCP tool definitions and request/response handling
2. **Service Layer**: Core business logic for decision analysis
3. **Infrastructure Layer**: Data persistence, external APIs, and utilities

Key requirements:
- Maintain existing MCP API compatibility
- Enable comprehensive unit testing of business logic
- Support future expansion (REST API, CLI tools, etc.)
- Improve code organization and maintainability
- Preserve performance characteristics

## Decision

We extract business logic into a dedicated **Service Layer** with clear responsibilities and dependencies:

### Service Layer Components

1. **DecisionService** - Core decision analysis orchestration
   - Session lifecycle management
   - Option and criterion management
   - Evaluation coordination
   - Result processing and storage

2. **ValidationService** - Centralized input validation
   - Session ID and format validation
   - Business rule enforcement
   - Prerequisite checking
   - Error message standardization

3. **ResponseService** - Standardized response formatting
   - Consistent response structure creation
   - Error response formatting
   - User-friendly message generation
   - Output formatting for different contexts

### Dependency Injection Architecture

- **Container Pattern**: `dependency_injection/container.py` manages service instances
- **Singleton Services**: Each service instantiated once and reused
- **Clear Dependencies**: Services declare their dependencies explicitly
- **Testing Support**: Easy mocking and stubbing for unit tests

### Layer Separation

```
┌─────────────────────────────────────┐
│          MCP Protocol Layer         │  ← __init__.py (tool handlers)
├─────────────────────────────────────┤
│           Service Layer             │  ← services/ (business logic)
├─────────────────────────────────────┤
│        Infrastructure Layer         │  ← models, orchestrator, backends
└─────────────────────────────────────┘
```

## Consequences

### Positive
- **Improved Testability**: Services can be unit tested independently of MCP protocol
- **Code Reusability**: Business logic available for CLI, REST API, or other interfaces
- **Clear Responsibilities**: Each service has a single, well-defined purpose
- **Easier Maintenance**: Changes to business logic isolated from protocol handling
- **Better Error Handling**: Centralized validation and error formatting
- **Dependency Clarity**: Explicit service dependencies enable better architecture understanding

### Negative
- **Increased Complexity**: More files and indirection compared to monolithic approach
- **Learning Curve**: Developers need to understand service layer concepts
- **Performance Overhead**: Additional method calls and object creation (minimal impact)

### Neutral
- **Configuration Management**: Dependency injection requires understanding of container pattern
- **Testing Strategy**: Unit tests now required at both service and integration levels

## Alternatives Considered

### Alternative 1: Monolithic Handlers with Helper Functions
**Description**: Keep all logic in `__init__.py` but extract helper functions for reusability.
**Pros**: Simpler structure, no dependency injection complexity
**Cons**: Still couples business logic to MCP protocol, limited testability
**Reason for rejection**: Doesn't solve fundamental testability and reusability issues

### Alternative 2: Domain-Driven Design with Repositories
**Description**: Full DDD approach with entities, value objects, repositories, and domain services.
**Pros**: Very clean domain modeling, excellent for complex business logic
**Cons**: Significant overhead for this application's scope, over-engineering
**Reason for rejection**: Too complex for the current requirements, would require major refactoring

### Alternative 3: Functional Programming Approach
**Description**: Pure functions for all business logic, avoiding object-oriented services.
**Pros**: Easier testing, no state management, functional composition
**Cons**: Doesn't fit well with existing OOP codebase, harder dependency management
**Reason for rejection**: Would require complete rewrite and doesn't integrate well with current architecture

## Implementation Notes

### Key Files
- `src/decision_matrix_mcp/services/__init__.py` - Service module exports
- `src/decision_matrix_mcp/services/decision_service.py` - Core business logic
- `src/decision_matrix_mcp/services/validation_service.py` - Input validation
- `src/decision_matrix_mcp/services/response_service.py` - Response formatting
- `src/decision_matrix_mcp/dependency_injection/container.py` - DI container

### Service Dependencies
```python
DecisionService
├── SessionManager (infrastructure)
└── DecisionOrchestrator (infrastructure)

ValidationService
└── SessionValidator (infrastructure)

ResponseService
└── DecisionFormatter (infrastructure)
```

### Migration Strategy
1. Extract services one at a time to minimize risk
2. Maintain backward compatibility during transition
3. Add comprehensive service-level unit tests
4. Update integration tests to use service layer
5. Refactor MCP handlers to use services

### Testing Strategy
- **Unit Tests**: Test each service in isolation with mocked dependencies
- **Integration Tests**: Test service interactions and MCP protocol compliance
- **End-to-End Tests**: Verify complete workflows through MCP interface

## References
- [Clean Architecture by Robert Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Dependency Injection Principles](https://martinfowler.com/articles/injection.html)
- [Service Layer Pattern](https://martinfowler.com/eaaCatalog/serviceLayer.html)
- [Testing Strategies for Service-Oriented Architecture](https://martinfowler.com/articles/microservice-testing/)

---
**Decision Date:** 2025-08-12
**Last Updated:** 2025-08-12
**Author(s):** Claude Code
**Review Status:** Approved
