"""
Shared pytest configuration and fixtures for decision-matrix-mcp tests.

This file contains:
- Common test fixtures used across test modules
- Test configuration and setup
- Markers for different test categories
- Environment variable handling for CI/local development
"""

import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test dependencies with fallbacks for CI environments
try:
    from decision_matrix_mcp import create_server_components
    from decision_matrix_mcp.session_manager import SessionManager

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    # Dependencies not available - tests will be skipped
    SessionManager = None
    create_server_components = None
    DEPENDENCIES_AVAILABLE = False


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for system interactions")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "slow: Tests that take longer than 30 seconds")
    config.addinivalue_line(
        "markers",
        "real_backends: Tests that require real LLM backend credentials",
    )
    config.addinivalue_line("markers", "concurrent: Tests that use concurrency/parallelism")
    config.addinivalue_line("markers", "memory: Tests that check memory usage patterns")


def pytest_collection_modifyitems(_config, items):
    """Modify test collection to add markers based on test location and content."""
    for item in items:
        # Add markers based on test file location
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Add slow marker for tests that likely take longer
        if any(
            keyword in item.name.lower()
            for keyword in ["sustained", "load", "concurrent", "scaling", "throughput", "benchmark"]
        ):
            item.add_marker(pytest.mark.slow)

        # Add concurrent marker for tests using asyncio or threading
        if any(
            keyword in item.name.lower()
            for keyword in ["concurrent", "parallel", "thread", "async"]
        ):
            item.add_marker(pytest.mark.concurrent)

        # Add memory marker for tests that check memory usage
        if any(keyword in item.name.lower() for keyword in ["memory", "leak", "growth", "cleanup"]):
            item.add_marker(pytest.mark.memory)

        # Add real_backends marker for tests requiring actual API credentials
        if (
            any(
                keyword in item.name.lower()
                for keyword in ["real_connection", "real_backend", "aws_bedrock_connection"]
            )
            and "mock" not in item.name.lower()
        ):
            item.add_marker(pytest.mark.real_backends)


@pytest.fixture()
def mock_context():
    """Provide a mock MCP context for testing."""
    from mcp.server.fastmcp import Context

    return Mock(spec=Context)


@pytest.fixture()
def clean_session_manager():
    """Provide a clean session manager for testing."""
    if not DEPENDENCIES_AVAILABLE:
        pytest.skip("Dependencies not available")

    manager = SessionManager(max_sessions=100, session_ttl_hours=1)
    yield manager
    # Cleanup after test
    for session_id in list(manager.list_active_sessions().keys()):
        manager.remove_session(session_id)


@pytest.fixture()
def server_components():
    """Provide server components for testing."""
    if not DEPENDENCIES_AVAILABLE:
        pytest.skip("Dependencies not available")

    components = create_server_components()
    yield components
    # Cleanup sessions after test
    session_manager = components.session_manager
    for session_id in list(session_manager.list_active_sessions().keys()):
        session_manager.remove_session(session_id)


@pytest.fixture()
def sample_decision_data():
    """Provide sample decision analysis data for testing."""
    return {
        "topic": "Choose a cloud database solution",
        "options": ["PostgreSQL on AWS RDS", "MongoDB Atlas", "Amazon DynamoDB"],
        "criteria": [
            {
                "name": "Performance",
                "description": "Query speed and throughput capabilities",
                "weight": 2.0,
            },
            {
                "name": "Cost",
                "description": "Total cost of ownership including licensing",
                "weight": 1.5,
            },
            {
                "name": "Scalability",
                "description": "Ability to handle growth in data and users",
                "weight": 1.8,
            },
            {
                "name": "Ease of Use",
                "description": "Developer experience and operational complexity",
                "weight": 1.2,
            },
        ],
    }


@pytest.fixture()
def mock_evaluation_results():
    """Provide mock evaluation results for testing."""
    return {
        "Performance": {
            "PostgreSQL on AWS RDS": (
                8.5,
                "Excellent performance for complex queries with proper indexing",
            ),
            "MongoDB Atlas": (
                7.2,
                "Good performance for document operations, varies with query patterns",
            ),
            "Amazon DynamoDB": (9.0, "Outstanding performance for key-value operations at scale"),
        },
        "Cost": {
            "PostgreSQL on AWS RDS": (7.0, "Moderate cost with predictable pricing model"),
            "MongoDB Atlas": (6.0, "Higher cost for enterprise features, complex pricing"),
            "Amazon DynamoDB": (8.0, "Cost-effective for moderate usage, can scale expensively"),
        },
        "Scalability": {
            "PostgreSQL on AWS RDS": (6.5, "Good vertical scaling, limited horizontal scaling"),
            "MongoDB Atlas": (8.5, "Excellent horizontal scaling with sharding"),
            "Amazon DynamoDB": (9.5, "Outstanding automatic scaling capabilities"),
        },
        "Ease of Use": {
            "PostgreSQL on AWS RDS": (8.0, "Familiar SQL interface, mature tooling ecosystem"),
            "MongoDB Atlas": (7.5, "Good developer tools, some learning curve for NoSQL"),
            "Amazon DynamoDB": (6.0, "Simple for basic use cases, complex for advanced features"),
        },
    }


@pytest.fixture(scope="session")
def aws_credentials_available():
    """Check if AWS credentials are available for real Bedrock testing."""
    return bool(
        os.environ.get("AWS_PROFILE")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")),
    )


@pytest.fixture(scope="session")
def litellm_credentials_available():
    """Check if LiteLLM credentials are available for real API testing."""
    return bool(
        os.environ.get("LITELLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY"),
    )


@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama is available locally for testing."""
    try:
        import asyncio

        import httpx

        async def check_ollama():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags")
                    return response.status_code == 200
            except Exception:
                return False

        return asyncio.run(check_ollama())
    except Exception:
        return False


@pytest.fixture()
def performance_test_config():
    """Provide configuration for performance tests."""
    return {
        "small_matrix": (3, 2),  # 3 options, 2 criteria
        "medium_matrix": (8, 5),  # 8 options, 5 criteria
        "large_matrix": (15, 8),  # 15 options, 8 criteria
        "max_response_time": 0.1,  # 100ms max response time
        "min_throughput": 10,  # 10 operations/second minimum
        "max_memory_per_session": 100 * 1024,  # 100KB per session
        "concurrent_sessions": 10,  # Number of concurrent sessions for testing
        "sustained_duration": 5,  # Duration for sustained load tests (seconds)
    }


@pytest.fixture()
def integration_test_config():
    """Provide configuration for integration tests."""
    return {
        "test_session_limit": 50,
        "test_criteria_limit": 10,
        "test_options_limit": 20,
        "mock_llm_delay": 0.01,  # 10ms mock LLM delay
        "timeout_seconds": 30,  # 30 second timeout for integration tests
    }


# Skip markers for optional dependencies
def pytest_runtest_setup(item):
    """Skip tests based on available dependencies and environment."""
    # Skip real backend tests if credentials not available
    if item.get_closest_marker("real_backends"):
        aws_available = bool(os.environ.get("AWS_PROFILE"))
        litellm_available = bool(
            os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )

        if not (aws_available or litellm_available):
            pytest.skip("Real backend credentials not available")

    # Skip slow tests in CI unless explicitly requested
    if (
        item.get_closest_marker("slow")
        and os.environ.get("CI")
        and not os.environ.get("RUN_SLOW_TESTS")
    ):
        pytest.skip("Slow tests skipped in CI (set RUN_SLOW_TESTS=1 to enable)")


# Performance test utilities
@pytest.fixture()
def benchmark_config():
    """Provide configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "warmup_rounds": 2,
        "disable_gc": True,
        "sort": "mean",
    }
