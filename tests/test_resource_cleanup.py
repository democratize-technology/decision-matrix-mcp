"""Test resource cleanup functionality."""

import contextlib
from unittest.mock import MagicMock, patch

from decision_matrix_mcp import ServerComponents
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestResourceCleanup:
    """Test proper resource cleanup on shutdown."""

    def test_orchestrator_cleanup(self):
        """Test that orchestrator cleanup properly clears resources."""
        orchestrator = DecisionOrchestrator()

        # Mock the bedrock client
        mock_client = MagicMock()
        orchestrator._bedrock_client = mock_client

        # Call cleanup
        orchestrator.cleanup()

        # Verify client reference was cleared
        assert orchestrator._bedrock_client is None

    def test_orchestrator_cleanup_thread_safe(self):
        """Test that cleanup is thread-safe."""
        import threading

        orchestrator = DecisionOrchestrator()

        # Set up multiple clients
        orchestrator._bedrock_client = MagicMock()

        # Track cleanup calls
        cleanup_count = 0
        cleanup_lock = threading.Lock()

        def cleanup_thread():
            nonlocal cleanup_count
            orchestrator.cleanup()
            with cleanup_lock:
                cleanup_count += 1

        # Run cleanup from multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=cleanup_thread)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify cleanup was successful
        assert orchestrator._bedrock_client is None
        assert cleanup_count == 5

    def test_server_components_cleanup(self):
        """Test that ServerComponents cleanup calls all component cleanups."""
        # Create a mock service container with the required methods
        mock_container = MagicMock()
        mock_orchestrator = MagicMock()
        mock_session_manager = MagicMock()

        # Set up container to return our mocked components
        mock_container.get_orchestrator.return_value = mock_orchestrator
        mock_container.get_session_manager.return_value = mock_session_manager

        components = ServerComponents(service_container=mock_container)

        # Call cleanup
        components.cleanup()

        # Verify container cleanup was called (which should clean up all components)
        mock_container.cleanup.assert_called_once()

    def test_cleanup_on_server_shutdown(self):
        """Test that cleanup is called during server shutdown."""
        import decision_matrix_mcp

        # Reset global state
        decision_matrix_mcp._server_components = None

        # Mock components
        mock_components = MagicMock()

        with patch("decision_matrix_mcp.create_server_components", return_value=mock_components):
            with patch("decision_matrix_mcp.mcp.run", side_effect=KeyboardInterrupt):
                # Run main and catch the expected exit
                with contextlib.suppress(SystemExit):
                    decision_matrix_mcp.main()

                # Verify cleanup was called
                mock_components.cleanup.assert_called_once()

    def test_cleanup_error_handling(self):
        """Test that cleanup errors are handled gracefully."""
        import decision_matrix_mcp

        # Reset global state
        decision_matrix_mcp._server_components = None

        # Mock components that raise during cleanup
        mock_components = MagicMock()
        mock_components.cleanup.side_effect = RuntimeError("Cleanup failed")

        with patch("decision_matrix_mcp.create_server_components", return_value=mock_components):
            with patch("decision_matrix_mcp.mcp.run", side_effect=KeyboardInterrupt):
                with patch("decision_matrix_mcp.logger") as mock_logger:
                    # Run main
                    with contextlib.suppress(SystemExit):
                        decision_matrix_mcp.main()

                    # Verify error was logged - logger.exception() just logs "Error during cleanup"
                    # The exception details are automatically included by the logging framework
                    mock_logger.exception.assert_any_call("Error during cleanup")

    def test_cleanup_idempotency(self):
        """Test that cleanup can be called multiple times safely."""
        orchestrator = DecisionOrchestrator()

        # Set a client
        orchestrator._bedrock_client = MagicMock()

        # Call cleanup multiple times
        orchestrator.cleanup()
        orchestrator.cleanup()
        orchestrator.cleanup()

        # Should still be None and not raise
        assert orchestrator._bedrock_client is None

    def test_no_cleanup_if_no_resources(self):
        """Test that cleanup works when no resources were allocated."""
        orchestrator = DecisionOrchestrator()

        # No client created
        assert orchestrator._bedrock_client is None

        # Cleanup should work without issues
        orchestrator.cleanup()

        # Still None
        assert orchestrator._bedrock_client is None
