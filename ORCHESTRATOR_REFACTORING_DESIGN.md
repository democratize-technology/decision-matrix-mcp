# Orchestrator Complexity Refactoring Design

## Executive Summary

This document presents a comprehensive architectural design for refactoring the `DecisionOrchestrator` class to address critical complexity issues. The current 627-line monolithic class violates Single Responsibility Principle and has methods with cyclomatic complexity of 15-20. This design targets a 70% complexity reduction through strategic application of design patterns while maintaining 100% API compatibility.

## Current Architecture Analysis

### Complexity Hotspots

**Primary Issues Identified:**

1. **`evaluate_options_across_criteria` (lines 189-227)**: 35 lines handling parallel execution, error collection, and result aggregation
2. **`_evaluate_single_option` (lines 229-304)**: 75 lines mixing prompt generation, CoT logic, backend routing, error handling, and response parsing  
3. **`_get_thread_response` (lines 411-456)**: 35 lines with exponential backoff and error classification
4. **Backend methods `_call_*` (lines 457-616)**: 160 lines of duplicated retry logic and error handling

### Architectural Problems

- **Mixed Responsibilities**: Single methods handle multiple concerns
- **Code Duplication**: Retry logic repeated across backends
- **Tight Coupling**: Backend specifics embedded in orchestrator
- **Testing Challenges**: Monolithic structure prevents isolated testing
- **Extensibility Issues**: Adding new backends requires modifying orchestrator

## Proposed Architecture

### Design Patterns Applied

#### 1. Strategy Pattern - Backend Abstraction

```python
from abc import ABC, abstractmethod
from typing import Protocol

class LLMBackend(ABC):
    """Abstract base for all LLM backend implementations"""
    
    @abstractmethod
    async def generate_response(self, request: EvaluationRequest) -> EvaluationResponse:
        """Generate response for evaluation request"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available"""
        pass

class BedrockBackend(LLMBackend):
    def __init__(self, client_factory: BedrockClientFactory):
        self.client_factory = client_factory
    
    async def generate_response(self, request: EvaluationRequest) -> EvaluationResponse:
        # Bedrock-specific implementation
        pass

class LiteLLMBackend(LLMBackend):
    async def generate_response(self, request: EvaluationRequest) -> EvaluationResponse:
        # LiteLLM-specific implementation  
        pass

class OllamaBackend(LLMBackend):
    async def generate_response(self, request: EvaluationRequest) -> EvaluationResponse:
        # Ollama-specific implementation
        pass
```

#### 2. Chain of Responsibility - Evaluation Pipeline

```python
class EvaluationHandler(ABC):
    """Base class for evaluation pipeline handlers"""
    
    def __init__(self, next_handler: Optional['EvaluationHandler'] = None):
        self.next_handler = next_handler
    
    @abstractmethod
    async def handle(self, command: EvaluationCommand) -> EvaluationResult:
        pass

class PromptGeneratorHandler(EvaluationHandler):
    """Generates evaluation prompts from thread and option"""
    
    async def handle(self, command: EvaluationCommand) -> EvaluationResult:
        prompt = self._generate_prompt(command.thread, command.option)
        command.context.prompt = prompt
        
        if self.next_handler:
            return await self.next_handler.handle(command)
        return EvaluationResult.success(command)

class CoTDecisionHandler(EvaluationHandler):
    """Decides whether to use Chain of Thought reasoning"""
    
    def __init__(self, reasoning_orchestrator: DecisionReasoningOrchestrator, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_orchestrator = reasoning_orchestrator
    
    async def handle(self, command: EvaluationCommand) -> EvaluationResult:
        if self._should_use_cot(command):
            command.context.use_cot = True
            command.context.cot_orchestrator = self.reasoning_orchestrator
        
        if self.next_handler:
            return await self.next_handler.handle(command)
        return EvaluationResult.success(command)

class BackendRouterHandler(EvaluationHandler):
    """Routes request to appropriate backend"""
    
    def __init__(self, backend_factory: BackendFactory, **kwargs):
        super().__init__(**kwargs)
        self.backend_factory = backend_factory
    
    async def handle(self, command: EvaluationCommand) -> EvaluationResult:
        backend = self.backend_factory.create_backend(
            command.thread.criterion.model_backend
        )
        
        try:
            response = await backend.generate_response(
                command.to_evaluation_request()
            )
            command.context.response = response
        except Exception as e:
            return EvaluationResult.error(command, e)
        
        if self.next_handler:
            return await self.next_handler.handle(command)
        return EvaluationResult.success(command)

class ResponseParserHandler(EvaluationHandler):
    """Parses LLM response into score and justification"""
    
    async def handle(self, command: EvaluationCommand) -> EvaluationResult:
        try:
            score, justification = self._parse_response(command.context.response)
            command.context.score = score
            command.context.justification = justification
        except Exception as e:
            return EvaluationResult.error(command, e)
        
        return EvaluationResult.success(command)
```

#### 3. Command Pattern - Evaluation Requests

```python
@dataclass
class EvaluationCommand:
    """Encapsulates an evaluation request"""
    thread: CriterionThread
    option: Option
    context: EvaluationContext = field(default_factory=EvaluationContext)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    
    def to_evaluation_request(self) -> EvaluationRequest:
        """Convert to backend-specific request format"""
        return EvaluationRequest(
            thread=self.thread,
            option=self.option,
            prompt=self.context.prompt,
            use_cot=self.context.use_cot,
            cot_orchestrator=self.context.cot_orchestrator
        )

@dataclass  
class EvaluationContext:
    """Mutable context passed through pipeline"""
    prompt: str = ""
    use_cot: bool = False
    cot_orchestrator: Optional[DecisionReasoningOrchestrator] = None
    response: Optional[EvaluationResponse] = None
    score: Optional[float] = None
    justification: str = ""

@dataclass
class EvaluationResult:
    """Result of evaluation pipeline"""
    success: bool
    command: EvaluationCommand
    error: Optional[Exception] = None
    
    @classmethod
    def success(cls, command: EvaluationCommand) -> 'EvaluationResult':
        return cls(success=True, command=command)
    
    @classmethod  
    def error(cls, command: EvaluationCommand, error: Exception) -> 'EvaluationResult':
        return cls(success=False, command=command, error=error)
```

#### 4. Factory Pattern - Backend Creation

```python
class BackendFactory:
    """Factory for creating LLM backend instances"""
    
    def __init__(self):
        self._backends: dict[ModelBackend, type[LLMBackend]] = {
            ModelBackend.BEDROCK: BedrockBackend,
            ModelBackend.LITELLM: LiteLLMBackend, 
            ModelBackend.OLLAMA: OllamaBackend,
        }
        self._instances: dict[ModelBackend, LLMBackend] = {}
    
    def create_backend(self, backend_type: ModelBackend) -> LLMBackend:
        """Create or return cached backend instance"""
        if backend_type not in self._instances:
            backend_class = self._backends[backend_type]
            self._instances[backend_type] = backend_class()
        
        return self._instances[backend_type]
    
    def validate_backend_availability(self, backend_type: ModelBackend) -> bool:
        """Check if backend dependencies are available"""
        try:
            backend = self.create_backend(backend_type)
            return backend.is_available()
        except Exception:
            return False
```

#### 5. Service Classes - Cross-Cutting Concerns

```python
class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def execute_with_retry(
        self, 
        operation: Callable[[], Awaitable[T]],
        error_classifier: ErrorClassifier
    ) -> T:
        """Execute operation with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await operation()
            except Exception as e:
                last_error = e
                
                if not error_classifier.is_retryable(e) or attempt == self.max_retries - 1:
                    break
                
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        raise last_error

class ErrorClassifier:
    """Classifies errors and determines retry eligibility"""
    
    def is_retryable(self, error: Exception) -> bool:
        """Determine if error should trigger retry"""
        error_str = str(error).lower()
        
        # Non-retryable errors
        non_retryable = [
            "api_key", "credentials", "not found", "invalid", 
            "unauthorized", "forbidden", "model not found"
        ]
        
        return not any(term in error_str for term in non_retryable)
    
    def get_user_message(self, error: Exception) -> str:
        """Get user-friendly error message"""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "quota" in error_str:
            return "API rate limit exceeded, please try again later"
        elif "api key" in error_str or "authentication" in error_str:
            return "API authentication failed, check your API key"
        elif "model" in error_str and "not found" in error_str:
            return "Model not available"
        else:
            return "Service temporarily unavailable"
```

### Refactored DecisionOrchestrator

```python
class DecisionOrchestrator:
    """Simplified orchestrator focused on coordination"""
    
    def __init__(
        self,
        backend_factory: BackendFactory,
        retry_handler: RetryHandler,
        reasoning_orchestrator: DecisionReasoningOrchestrator,
        use_cot: bool = True
    ):
        self.backend_factory = backend_factory
        self.retry_handler = retry_handler
        self.evaluation_pipeline = self._build_evaluation_pipeline(
            reasoning_orchestrator, use_cot
        )
    
    def _build_evaluation_pipeline(
        self, 
        reasoning_orchestrator: DecisionReasoningOrchestrator,
        use_cot: bool
    ) -> EvaluationHandler:
        """Build the evaluation pipeline chain"""
        return PromptGeneratorHandler(
            next_handler=CoTDecisionHandler(
                reasoning_orchestrator=reasoning_orchestrator,
                next_handler=BackendRouterHandler(
                    backend_factory=self.backend_factory,
                    next_handler=ResponseParserHandler()
                )
            )
        )
    
    async def evaluate_options_across_criteria(
        self, threads: dict[str, CriterionThread], options: list[Option]
    ) -> dict[str, dict[str, tuple[float | None, str]]]:
        """Orchestrate parallel evaluation across all criterion-option pairs"""
        
        # Create evaluation tasks
        tasks = []
        task_metadata = []
        
        for criterion_name, thread in threads.items():
            for option in options:
                task = self._evaluate_option_pair(thread, option)
                tasks.append(task)
                task_metadata.append((criterion_name, option.name))
        
        logger.info(
            f"Starting parallel evaluation of {len(options)} options "
            f"across {len(threads)} criteria"
        )
        
        # Execute all evaluations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect and organize results
        return self._organize_results(results, task_metadata)
    
    async def _evaluate_option_pair(
        self, thread: CriterionThread, option: Option
    ) -> tuple[float | None, str]:
        """Evaluate a single thread-option pair through the pipeline"""
        command = EvaluationCommand(thread=thread, option=option)
        
        try:
            result = await self.evaluation_pipeline.handle(command)
            
            if result.success:
                # Add assistant message to thread conversation
                response_text = f"SCORE: {result.command.context.score or 'NO_RESPONSE'}\nJUSTIFICATION: {result.command.context.justification}"
                thread.add_message("assistant", response_text)
                
                return (result.command.context.score, result.command.context.justification)
            else:
                logger.error(f"Evaluation failed for {option.name}: {result.error}")
                return (None, f"Error: {str(result.error)}")
                
        except Exception as e:
            logger.exception(f"Unexpected error evaluating {option.name}")
            return (None, "Evaluation failed due to an unexpected error")
    
    def _organize_results(
        self, 
        results: list[Any], 
        task_metadata: list[tuple[str, str]]
    ) -> dict[str, dict[str, tuple[float | None, str]]]:
        """Organize raw results into criterion -> option -> (score, justification) structure"""
        evaluation_results: dict[str, dict[str, tuple[float | None, str]]] = {}
        
        for i, result in enumerate(results):
            criterion_name, option_name = task_metadata[i]
            
            if criterion_name not in evaluation_results:
                evaluation_results[criterion_name] = {}
            
            if isinstance(result, Exception):
                logger.error(f"Error evaluating {option_name} for {criterion_name}: {result}")
                evaluation_results[criterion_name][option_name] = (None, f"Error: {str(result)}")
            elif isinstance(result, tuple):
                evaluation_results[criterion_name][option_name] = result
            else:
                logger.error(f"Unexpected result type for {option_name}/{criterion_name}: {type(result)}")
                evaluation_results[criterion_name][option_name] = (None, "Error: Unexpected result type")
        
        return evaluation_results
    
    async def test_bedrock_connection(self) -> dict[str, Any]:
        """Test Bedrock backend connectivity"""
        backend = self.backend_factory.create_backend(ModelBackend.BEDROCK)
        return await backend.test_connection()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Cleanup managed by individual backend instances
        pass
```

## Migration Strategy

### Phase 1: Extract Backend Strategies (Week 1)
**Goal**: Separate backend-specific logic into strategy pattern

**Tasks**:
1. Create `LLMBackend` interface and implementations
2. Move logic from `_call_bedrock`, `_call_litellm`, `_call_ollama` 
3. Create `BackendFactory` with availability checks
4. Update orchestrator to use factory (keeping existing methods as facades)
5. Comprehensive testing of backend isolation

**Success Criteria**:
- All existing tests pass
- Backend logic is isolated and testable
- No change to public API

### Phase 2: Create Evaluation Pipeline (Week 2)  
**Goal**: Extract evaluation concerns into chain of responsibility

**Tasks**:
1. Implement evaluation handler base class and concrete handlers
2. Create `EvaluationCommand` and related data structures
3. Build pipeline in orchestrator initialization
4. Update `_evaluate_single_option` to use pipeline (keeping as facade)
5. Test pipeline with all evaluation scenarios

**Success Criteria**:
- Evaluation logic is modular and testable
- CoT integration works seamlessly
- Response parsing is isolated

### Phase 3: Extract Retry and Error Handling (Week 3)
**Goal**: Centralize retry logic and error classification

**Tasks**:
1. Create `RetryHandler` and `ErrorClassifier` services
2. Replace inline retry logic in backends
3. Standardize error handling across all backends
4. Update `_get_thread_response` to use services
5. Test error scenarios and retry behavior

**Success Criteria**:
- Consistent error handling across backends
- Configurable retry behavior
- Better error messages for users

### Phase 4: Finalize Orchestrator Refactoring (Week 4)
**Goal**: Complete transformation to modular architecture

**Tasks**:
1. Refactor `evaluate_options_across_criteria` to use new pipeline directly
2. Remove facade methods (`_evaluate_single_option`, `_get_thread_response`, `_call_*`)
3. Clean up unused imports and methods
4. Update documentation and examples
5. Performance benchmarking and optimization

**Success Criteria**:
- Code complexity reduced by >70%
- Performance maintained or improved
- Full test coverage of new architecture

## Testing Strategy

### Unit Testing

**Backend Testing**:
```python
class TestBedrockBackend:
    @pytest.fixture
    def mock_bedrock_client(self):
        return Mock()
    
    @pytest.fixture
    def backend(self, mock_bedrock_client):
        factory = Mock()
        factory.create_client.return_value = mock_bedrock_client
        return BedrockBackend(factory)
    
    async def test_generate_response_success(self, backend, mock_bedrock_client):
        # Test successful response generation
        mock_bedrock_client.converse.return_value = {...}
        
        request = EvaluationRequest(...)
        response = await backend.generate_response(request)
        
        assert response.text == "expected response"
        mock_bedrock_client.converse.assert_called_once()
```

**Pipeline Testing**:
```python
class TestEvaluationPipeline:
    @pytest.fixture
    def mock_handlers(self):
        return [Mock(spec=EvaluationHandler) for _ in range(4)]
    
    async def test_pipeline_execution(self, mock_handlers):
        # Test that command flows through all handlers
        handler1, handler2, handler3, handler4 = mock_handlers
        
        # Build pipeline
        handler1.next_handler = handler2
        handler2.next_handler = handler3  
        handler3.next_handler = handler4
        
        command = EvaluationCommand(...)
        await handler1.handle(command)
        
        # Verify all handlers were called
        for handler in mock_handlers:
            handler.handle.assert_called_once_with(command)
```

### Integration Testing

**End-to-End Evaluation**:
```python
class TestEvaluationIntegration:
    async def test_full_evaluation_flow(self):
        # Test complete evaluation from command to result
        orchestrator = DecisionOrchestrator(...)
        threads = {...}
        options = [...]
        
        results = await orchestrator.evaluate_options_across_criteria(threads, options)
        
        assert len(results) == len(threads)
        for criterion_results in results.values():
            assert len(criterion_results) == len(options)
```

### Performance Testing

**Benchmarking Framework**:
```python
class TestPerformance:
    @pytest.mark.benchmark
    async def test_evaluation_latency(self, benchmark):
        # Measure evaluation latency before/after refactoring
        orchestrator = DecisionOrchestrator(...)
        
        result = await benchmark.pedantic(
            orchestrator.evaluate_options_across_criteria,
            args=(threads, options),
            iterations=10,
            rounds=3
        )
        
        assert result is not None
```

## Performance Considerations

### Overhead Analysis

**Abstraction Layer Impact**:
- Strategy pattern: +1 virtual method call per backend operation
- Chain of responsibility: +3-4 method calls per evaluation  
- Command pattern: +object creation overhead per evaluation

**Mitigation Strategies**:
- Cache backend instances in factory (singleton pattern)
- Pre-build pipeline during orchestrator initialization
- Use object pooling for frequently created objects
- Profile critical paths and optimize hot spots

### Memory Optimization

- Backend instances are stateless and can be singletons
- Reuse `EvaluationCommand` objects where possible
- Clear conversation history in threads after evaluation if memory is constrained
- Use `__slots__` for frequently created data classes

### Parallel Processing Benefits

- Cleaner separation enables better parallel evaluation optimization
- Backend-specific concurrency limits can be implemented
- Different backends can have different timeout strategies
- Error isolation prevents one backend failure from affecting others

## Benefits Analysis

### Code Quality Improvements

**Before Refactoring**:
- Single 627-line class with multiple responsibilities
- Methods with 35-75 lines and cyclomatic complexity 15-20
- Duplicated retry logic across backends
- Difficult to test individual concerns

**After Refactoring**:
- 6 focused components with single responsibilities
- Methods with 10-20 lines and cyclomatic complexity 3-5
- Centralized retry and error handling
- Each component independently testable

### Maintainability Benefits

- **Easier Debugging**: Clear component boundaries help isolate issues
- **Simpler Code Reviews**: Smaller, focused changes
- **Better Documentation**: Each component has clear purpose
- **Reduced Coupling**: Dependencies are explicit and injectable

### Extensibility Improvements

**Adding New Backends**: 
1. Implement `LLMBackend` interface
2. Register with `BackendFactory`
3. No changes to orchestrator or evaluation logic

**New Evaluation Features**:
- Add caching handler to pipeline
- Implement validation handler  
- Add metrics collection handler
- Insert A/B testing handler

### Testing Improvements

- **Unit Tests**: Each component can be tested in isolation
- **Integration Tests**: Clear interfaces enable better integration testing
- **Mock Strategy**: Dependencies can be easily mocked
- **Test Coverage**: Granular testing increases overall coverage

## Risk Mitigation

### API Compatibility

- **Public API Unchanged**: All public methods maintain exact signatures
- **Behavioral Compatibility**: Same inputs produce same outputs
- **Error Handling**: Error types and messages remain consistent

### Performance Risks

- **Benchmarking**: Measure performance before and after each phase
- **Rollback Plan**: Each phase can be independently rolled back
- **Optimization**: Profile and optimize any performance regressions

### Migration Risks

- **Incremental Changes**: Small, testable changes in each phase
- **Test Coverage**: Comprehensive tests before and after each change
- **Code Review**: Thorough review of each pattern implementation
- **Monitoring**: Track error rates and performance during migration

## Success Metrics

### Quantitative Metrics

- **Complexity Reduction**: Target 70% reduction in cyclomatic complexity
- **Line Count**: Reduce individual method lengths by 60-80%
- **Test Coverage**: Achieve >90% test coverage
- **Performance**: Maintain or improve evaluation latency

### Qualitative Metrics

- **Code Readability**: Each component has clear, single responsibility
- **Maintainability**: Time to add new backends reduced from days to hours
- **Debuggability**: Issues can be isolated to specific components
- **Extensibility**: New features can be added without modifying existing code

## Conclusion

This refactoring design transforms the monolithic `DecisionOrchestrator` into a modular, extensible architecture that follows SOLID principles and established design patterns. The phased migration approach ensures zero breaking changes while dramatically improving code quality, testability, and maintainability.

The investment in this refactoring will pay dividends in:
- Reduced development time for new features
- Improved system reliability and error handling  
- Better testing and debugging capabilities
- Easier onboarding for new developers
- Enhanced scalability for future requirements

The architecture positions the decision matrix system for long-term success with a clean, professional codebase that follows industry best practices.