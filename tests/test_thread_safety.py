"""Test thread safety of server components initialization."""

import threading
import time
from unittest.mock import patch

import pytest

import decision_matrix_mcp
from decision_matrix_mcp import (
    ServerComponents,
    get_server_components,
    initialize_server_components,
)


class TestThreadSafetyInitialization:
    """Test thread-safe initialization of server components."""

    def test_concurrent_initialization(self):
        """Test that concurrent threads safely initialize server components."""
        # Reset global state for this test
        import decision_matrix_mcp

        decision_matrix_mcp._server_components = None

        # Track initialization calls
        initialization_count = 0
        initialization_lock = threading.Lock()

        original_create = decision_matrix_mcp.create_server_components

        def tracked_create():
            nonlocal initialization_count
            with initialization_lock:
                initialization_count += 1
                time.sleep(0.01)  # Simulate slow initialization
            return original_create()

        with patch("decision_matrix_mcp.create_server_components", tracked_create):
            # Start multiple threads trying to initialize
            threads = []
            results = []
            errors = []

            def init_thread():
                try:
                    result = get_server_components()
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            # Create 10 threads
            for _ in range(10):
                thread = threading.Thread(target=init_thread)
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify no errors occurred
            assert len(errors) == 0, f"Threads encountered errors: {errors}"

            # Verify all threads got the same instance
            assert len(results) == 10
            first_result = results[0]
            for result in results:
                assert result is first_result

            # Verify initialization happened only once
            assert initialization_count == 1

    def test_lazy_initialization(self):
        """Test that components are initialized lazily on first access."""
        # Reset global state
        import decision_matrix_mcp

        decision_matrix_mcp._server_components = None

        # Access should trigger initialization
        components = get_server_components()
        assert components is not None
        assert isinstance(components, ServerComponents)

        # Second access should return same instance
        components2 = get_server_components()
        assert components2 is components

    def test_explicit_initialization(self):
        """Test that explicit initialization works correctly."""
        # Reset global state
        import decision_matrix_mcp

        decision_matrix_mcp._server_components = None

        # Explicit initialization
        initialize_server_components()

        # Verify components are initialized
        components = get_server_components()
        assert components is not None
        assert isinstance(components, ServerComponents)

    def test_double_checked_locking(self):
        """Test that double-checked locking pattern works correctly."""
        # Reset global state
        import decision_matrix_mcp

        decision_matrix_mcp._server_components = None

        # Track whether lock was acquired using a different approach
        create_calls = []
        original_create = decision_matrix_mcp.create_server_components

        def tracked_create():
            create_calls.append(time.time())
            return original_create()

        with patch("decision_matrix_mcp.create_server_components", tracked_create):
            # First call should create components
            components1 = get_server_components()
            assert len(create_calls) == 1

            # Second call should not create again (fast path)
            components2 = get_server_components()
            assert len(create_calls) == 1
            assert components2 is components1

    def test_initialization_error_handling(self):
        """Test that initialization errors are properly handled."""
        # Reset global state
        import decision_matrix_mcp

        decision_matrix_mcp._server_components = None

        # Make initialization fail
        with patch(
            "decision_matrix_mcp.create_server_components", side_effect=RuntimeError("Init failed")
        ):
            with pytest.raises(RuntimeError, match="Init failed"):
                get_server_components()

            # Verify subsequent calls also fail (no partial initialization)
            with pytest.raises(RuntimeError, match="Init failed"):
                get_server_components()
