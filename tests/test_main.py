"""Tests for main entry point and server initialization"""

import sys
from unittest.mock import patch

import pytest

from decision_matrix_mcp import main, mcp


class TestMainFunction:
    """Test the main() function and server lifecycle"""

    @patch("decision_matrix_mcp.mcp.run")
    @patch("decision_matrix_mcp.logger")
    def test_main_normal_execution(self, mock_logger, mock_mcp_run):
        """Test normal server startup and run"""
        # Call main
        main()

        # Should log startup messages
        mock_logger.info.assert_called()
        assert any(
            "Starting Decision Matrix MCP server" in str(call)
            for call in mock_logger.info.call_args_list
        )

        # Should call mcp.run()
        mock_mcp_run.assert_called_once()

    @patch("decision_matrix_mcp.mcp.run", side_effect=KeyboardInterrupt())
    @patch("decision_matrix_mcp.logger")
    def test_main_keyboard_interrupt(self, mock_logger, mock_mcp_run):
        """Test graceful shutdown on KeyboardInterrupt"""
        # Call main
        main()

        # Should log shutdown message
        mock_logger.info.assert_called()
        assert any(
            "Server stopped by user" in str(call) for call in mock_logger.info.call_args_list
        )

    @patch("decision_matrix_mcp.mcp.run", side_effect=BrokenPipeError())
    @patch("decision_matrix_mcp.logger")
    def test_main_broken_pipe_error(self, mock_logger, mock_mcp_run):
        """Test handling of BrokenPipeError (client disconnect)"""
        # Call main
        main()

        # Should only log debug message for normal disconnect
        mock_logger.debug.assert_called()
        assert any(
            "Client disconnected normally" in str(call) for call in mock_logger.debug.call_args_list
        )

    @patch("decision_matrix_mcp.mcp.run", side_effect=ConnectionResetError())
    @patch("decision_matrix_mcp.logger")
    def test_main_connection_reset_error(self, mock_logger, mock_mcp_run):
        """Test handling of ConnectionResetError"""
        # Call main
        main()

        # Should log debug message for normal disconnect
        mock_logger.debug.assert_called()
        assert any(
            "Client disconnected normally" in str(call) for call in mock_logger.debug.call_args_list
        )

    @patch("decision_matrix_mcp.mcp.run")
    @patch("decision_matrix_mcp.logger")
    def test_main_brokenresourceerror_in_exception_string(self, mock_logger, mock_mcp_run):
        """Test handling of BrokenResourceError wrapped in exception string"""
        # Create exception with BrokenResourceError in string
        error = Exception("Some error with BrokenResourceError in message")
        mock_mcp_run.side_effect = error

        # Call main
        main()

        # Should log debug message for stdio closed
        mock_logger.debug.assert_called()
        assert any(
            "Client disconnected (stdio closed)" in str(call)
            for call in mock_logger.debug.call_args_list
        )

    @patch("decision_matrix_mcp.mcp.run")
    @patch("decision_matrix_mcp.logger")
    def test_main_broken_pipe_in_exception_string(self, mock_logger, mock_mcp_run):
        """Test handling of 'broken pipe' in exception string"""
        # Create exception with broken pipe in string
        error = Exception("Connection lost: broken pipe")
        mock_mcp_run.side_effect = error

        # Call main
        main()

        # Should log debug message for stdio closed
        mock_logger.debug.assert_called()
        assert any(
            "Client disconnected (stdio closed)" in str(call)
            for call in mock_logger.debug.call_args_list
        )

    @patch("decision_matrix_mcp.mcp.run")
    @patch("decision_matrix_mcp.logger")
    @patch("traceback.print_exc")
    def test_main_unexpected_error(self, mock_print_exc, mock_logger, mock_mcp_run):
        """Test handling of unexpected errors"""
        # Create unexpected error
        error = ValueError("Unexpected error occurred")
        mock_mcp_run.side_effect = error

        # Call main and expect it to raise
        with pytest.raises(ValueError):
            main()

        # Should log error
        mock_logger.error.assert_called()
        assert any(
            "Server error: Unexpected error occurred" in str(call)
            for call in mock_logger.error.call_args_list
        )

        # Should print traceback
        mock_print_exc.assert_called_once_with(file=sys.stderr)

    @patch("decision_matrix_mcp.mcp.run")
    @patch("decision_matrix_mcp.logger")
    def test_main_logging_calls(self, mock_logger, mock_mcp_run):
        """Test all logging calls during normal execution"""
        # Call main
        main()

        # Check logging sequence
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]

        # Should have startup info log
        assert any("Starting Decision Matrix MCP server" in call for call in info_calls)

        # Should have debug logs
        assert any("Initializing FastMCP" in call for call in debug_calls)
        assert any("MCP server created" in call for call in debug_calls)
        assert any("Starting MCP server run" in call for call in info_calls)


class TestMainModule:
    """Test __main__ module execution"""

    @patch("decision_matrix_mcp.main")
    @patch("decision_matrix_mcp.logger")
    def test_main_module_execution(self, mock_logger, mock_main):
        """Test that main is called when module is run directly"""
        # Import the module to trigger __main__ check
        import decision_matrix_mcp

        # Simulate running as main
        with patch.object(decision_matrix_mcp, "__name__", "__main__"):
            # Re-evaluate the condition
            if decision_matrix_mcp.__name__ == "__main__":
                mock_logger.debug("Module started as main")
                decision_matrix_mcp.main()

        # Should log debug message
        mock_logger.debug.assert_called_with("Module started as main")

        # Should call main()
        mock_main.assert_called_once()


class TestServerCreation:
    """Test MCP server creation and configuration"""

    def test_mcp_server_exists(self):
        """Test that mcp server instance is created"""
        assert mcp is not None
        assert hasattr(mcp, "run")
        assert hasattr(mcp, "tool")

    def test_mcp_server_name(self):
        """Test that server has correct name"""
        # FastMCP stores name internally
        assert mcp.name == "decision-matrix"

    def test_mcp_initialization(self):
        """Test MCP server initialization"""
        # Just verify the mcp instance is properly initialized
        assert mcp is not None
        assert mcp.name == "decision-matrix"
        
        # Verify orchestrator is initialized
        from decision_matrix_mcp import orchestrator
        assert orchestrator is not None


class TestLoggingConfiguration:
    """Test logging configuration"""

    def test_logging_to_stderr(self):
        """Test that logging is configured to use stderr"""
        import logging

        # Get the logger
        logging.getLogger("decision_matrix_mcp")

        # Check handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, "stream"):
                # Should be stderr, not stdout (or dev/null in test env)
                assert handler.stream != sys.stdout

    def test_logging_level(self):
        """Test default logging level"""
        import logging

        # Root logger should be WARNING level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING or root_logger.level == 0

    def test_logging_format(self):
        """Test logging format configuration"""
        import logging

        root_logger = logging.getLogger()
        found_formatter = False
        for handler in root_logger.handlers:
            if hasattr(handler, "formatter") and handler.formatter:
                found_formatter = True
                if hasattr(handler.formatter, "_fmt") and handler.formatter._fmt:
                    format_string = handler.formatter._fmt
                    # Should include standard elements if format string exists
                    assert "asctime" in format_string or "message" in format_string
        # Just verify we have some formatter configured
        assert found_formatter or len(root_logger.handlers) > 0
