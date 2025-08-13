# Decision Matrix MCP - Test Suite

This directory contains comprehensive tests for the decision-matrix-mcp project, including unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── conftest.py                     # Shared pytest configuration and fixtures
├── README.md                       # This file
├── integration/                    # Integration tests
│   ├── __init__.py
│   ├── test_mcp_protocol.py       # End-to-end MCP protocol tests
│   ├── test_backend_integration.py # LLM backend integration tests
│   └── test_concurrent_sessions.py # Concurrent session management tests
├── performance/                    # Performance and benchmark tests
│   ├── __init__.py
│   ├── test_evaluation_performance.py    # Evaluation scaling and throughput
│   ├── test_session_performance.py       # Session management performance
│   └── test_backend_performance.py       # Backend response time and throughput
└── [existing unit tests...]       # Existing unit test files
```

## Test Categories

### Unit Tests (Existing)
- **Location**: `tests/test_*.py` (root level)
- **Purpose**: Test individual components and functions
- **Scope**: Models, session management, validation, formatting
- **Run with**: `pytest tests/test_*.py`

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test system interactions and end-to-end workflows
- **Scope**: MCP protocol, backend integration, concurrent operations
- **Run with**: `pytest tests/integration/`

### Performance Tests
- **Location**: `tests/performance/`
- **Purpose**: Measure performance, detect regressions, establish baselines
- **Scope**: Evaluation scaling, memory usage, throughput, response times
- **Run with**: `pytest tests/performance/`

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for system interactions
- `@pytest.mark.performance` - Performance and benchmark tests
- `@pytest.mark.slow` - Tests that take longer than 30 seconds
- `@pytest.mark.real_backends` - Tests requiring real LLM backend credentials
- `@pytest.mark.concurrent` - Tests using concurrency/parallelism
- `@pytest.mark.memory` - Tests checking memory usage patterns

## Running Tests

### Quick Test Run (Unit Tests Only)
```bash
# Run fast unit tests only
pytest -m "not slow"

# Run with coverage
pytest --cov=decision_matrix_mcp --cov-report=html tests/
```

### Integration Tests
```bash
# Run all integration tests
pytest tests/integration/

# Run specific integration test
pytest tests/integration/test_mcp_protocol.py

# Run integration tests excluding real backends
pytest tests/integration/ -m "not real_backends"
```

### Performance Tests
```bash
# Run all performance tests
pytest tests/performance/

# Run performance tests with output (to see performance metrics)
pytest tests/performance/ -s

# Run specific performance test
pytest tests/performance/test_evaluation_performance.py::TestEvaluationPerformanceScaling::test_evaluation_performance_scaling
```

### Complete Test Suite
```bash
# Run everything (may take several minutes)
pytest

# Run everything in parallel (requires pytest-xdist)
pytest -n auto

# Run with timeout protection
pytest --timeout=300
```

### Test Selection Examples
```bash
# Only fast tests
pytest -m "not slow"

# Only slow/comprehensive tests
pytest -m "slow"

# Only concurrent/parallel tests
pytest -m "concurrent"

# Only memory-related tests
pytest -m "memory"

# Integration tests without real backend dependencies
pytest -m "integration and not real_backends"

# Performance tests for evaluation only
pytest tests/performance/test_evaluation_performance.py
```

## Environment Configuration

### Required Environment Variables
None - tests use mocks by default for maximum compatibility.

### Optional Environment Variables (for real backend testing)

#### AWS Bedrock Testing
```bash
export AWS_PROFILE=your-profile
# OR
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
```

#### LiteLLM Testing
```bash
export LITELLM_API_KEY=your-key
# OR
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
```

#### Ollama Testing
```bash
export OLLAMA_HOST=http://localhost:11434
```

### CI Environment Variables
```bash
export CI=true                    # Enables CI-specific behavior
export RUN_SLOW_TESTS=1          # Enables slow tests in CI
export PYTEST_TIMEOUT=600        # Global test timeout (seconds)
```

## Test Fixtures

Common fixtures available in all tests:

- `mock_context` - Mock MCP context for handler testing
- `clean_session_manager` - Fresh session manager with cleanup
- `server_components` - Complete server components with cleanup
- `sample_decision_data` - Sample decision analysis data
- `mock_evaluation_results` - Mock LLM evaluation results
- `performance_test_config` - Configuration for performance tests
- `integration_test_config` - Configuration for integration tests

## Performance Test Interpretation

### Evaluation Performance
- **Matrix Size Scaling**: Tests evaluate performance across different matrix sizes (2x2 to 20x10)
- **Expected Metrics**:
  - Small matrices (2x2): < 10ms
  - Medium matrices (10x5): < 50ms
  - Large matrices (20x10): < 200ms
  - Throughput: > 50 evaluations/second

### Session Management Performance
- **Session Creation**: < 10ms per session
- **Session Retrieval**: < 1ms per lookup
- **Session Listing**: < 5ms for up to 200 sessions
- **Memory Usage**: < 50KB per session
- **Cleanup**: > 100 sessions/second removal rate

### Backend Performance
- **Response Times**:
  - Mock Bedrock: ~100ms (simulated)
  - Mock LiteLLM: ~50ms (simulated)
  - Mock Ollama: ~200ms (simulated)
- **Throughput**: > 10 requests/second per backend
- **Concurrent Load**: Handle 10+ concurrent requests

## Continuous Integration

### GitHub Actions Configuration
```yaml
# Example .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest -m "not slow and not real_backends" --cov
      - run: pytest -m "slow" --timeout=600  # Slow tests with timeout
```

### Local Development
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests before committing
pytest -m "not slow"

# Run full test suite before releases
pytest --timeout=600
```

## Troubleshooting

### Common Issues

#### Slow Test Performance
```bash
# Check if running in debug mode
pytest --tb=no -q

# Use parallel execution
pytest -n auto

# Skip slow tests during development
pytest -m "not slow"
```

#### Memory Issues in Performance Tests
```bash
# Run with memory profiling
python -m tracemalloc
pytest tests/performance/test_session_performance.py::test_session_memory_growth_patterns -s

# Check for memory leaks
pytest tests/performance/ -k "memory" -s
```

#### Integration Test Failures
```bash
# Check backend availability
pytest tests/integration/test_backend_integration.py::test_backend_availability_check -s

# Run with verbose output
pytest tests/integration/ -v -s

# Skip real backend tests
pytest tests/integration/ -m "not real_backends"
```

#### Timeout Issues
```bash
# Increase timeout for slow systems
pytest --timeout=600

# Run specific slow tests
pytest -m "slow" --timeout=900

# Debug hanging tests
pytest --pdb-trace
```

### Performance Baseline Updates

When system performance changes significantly:

1. Run baseline establishment:
   ```bash
   pytest tests/performance/test_evaluation_performance.py::test_baseline_performance_metrics -s
   ```

2. Review performance metrics in output

3. Update expected performance thresholds in test files if changes are intentional

4. Document performance changes in CHANGELOG.md

## Contributing

When adding new tests:

1. **Choose the right category**: Unit, integration, or performance
2. **Use appropriate markers**: Add `@pytest.mark.*` decorators
3. **Follow naming conventions**: `test_*` functions, `Test*` classes
4. **Add docstrings**: Explain what the test verifies
5. **Use existing fixtures**: Leverage shared test fixtures
6. **Consider CI impact**: Mark slow tests appropriately
7. **Document new patterns**: Update this README if needed

### Test Writing Guidelines

1. **Test Isolation**: Each test should be independent
2. **Deterministic Results**: Avoid flaky tests with proper mocking
3. **Clear Assertions**: Use descriptive assertion messages
4. **Performance Bounds**: Set realistic performance expectations
5. **Error Handling**: Test both success and failure cases
6. **Documentation**: Include docstrings explaining test purpose

For more detailed information, see the main project documentation and individual test file docstrings.
