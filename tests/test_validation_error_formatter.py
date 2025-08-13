"""Tests for ValidationErrorFormatter service in validation_decorators.py"""

from unittest.mock import Mock

from decision_matrix_mcp.formatting import DecisionFormatter
from decision_matrix_mcp.validation_decorators import ValidationErrorFormatter


class TestValidationErrorFormatter:
    """Test ValidationErrorFormatter service functionality"""

    def setup_method(self):
        """Reset formatter before each test"""
        ValidationErrorFormatter.reset()

    def test_format_error_without_initialization(self):
        """Test error formatting when formatter is not initialized (fallback mode)"""
        # Without context
        result = ValidationErrorFormatter.format_error("Test error message")
        assert result == "❌ Test error message"

        # With context
        result = ValidationErrorFormatter.format_error("Test error message", "Validation error")
        assert result == "❌ Validation error: Test error message"

    def test_format_error_with_initialization(self):
        """Test error formatting when formatter is properly initialized"""
        # Create a mock formatter
        mock_formatter = Mock(spec=DecisionFormatter)
        mock_formatter.format_error.return_value = "Formatted error message"

        # Initialize the service
        ValidationErrorFormatter.initialize(mock_formatter)

        # Test formatting
        result = ValidationErrorFormatter.format_error("Test error", "Test context")

        # Verify formatter was called correctly
        mock_formatter.format_error.assert_called_once_with("Test error", "Test context")
        assert result == "Formatted error message"

    def test_format_error_with_initialization_no_context(self):
        """Test error formatting with formatter but no context"""
        mock_formatter = Mock(spec=DecisionFormatter)
        mock_formatter.format_error.return_value = "Formatted error"

        ValidationErrorFormatter.initialize(mock_formatter)

        result = ValidationErrorFormatter.format_error("Test error")

        mock_formatter.format_error.assert_called_once_with("Test error", "")
        assert result == "Formatted error"

    def test_reset_functionality(self):
        """Test that reset clears the formatter"""
        mock_formatter = Mock(spec=DecisionFormatter)
        ValidationErrorFormatter.initialize(mock_formatter)

        # Verify formatter is set
        result = ValidationErrorFormatter.format_error("Test")
        mock_formatter.format_error.assert_called_once()

        # Reset and verify fallback is used
        ValidationErrorFormatter.reset()
        result = ValidationErrorFormatter.format_error("Test error")
        assert result == "❌ Test error"

    def test_multiple_initializations(self):
        """Test that multiple initializations replace the formatter"""
        mock_formatter1 = Mock(spec=DecisionFormatter)
        mock_formatter1.format_error.return_value = "Formatter 1"

        mock_formatter2 = Mock(spec=DecisionFormatter)
        mock_formatter2.format_error.return_value = "Formatter 2"

        # Initialize with first formatter
        ValidationErrorFormatter.initialize(mock_formatter1)
        result = ValidationErrorFormatter.format_error("Test")
        assert result == "Formatter 1"

        # Initialize with second formatter
        ValidationErrorFormatter.initialize(mock_formatter2)
        result = ValidationErrorFormatter.format_error("Test")
        assert result == "Formatter 2"

        # Verify only the second formatter was called
        mock_formatter1.format_error.assert_called_once()
        mock_formatter2.format_error.assert_called_once()

    def test_thread_safety_simulation(self):
        """Test that the service handles concurrent access safely"""
        mock_formatter = Mock(spec=DecisionFormatter)
        mock_formatter.format_error.return_value = "Thread-safe result"

        ValidationErrorFormatter.initialize(mock_formatter)

        # Simulate multiple concurrent calls
        results = []
        for i in range(10):
            result = ValidationErrorFormatter.format_error(f"Message {i}", f"Context {i}")
            results.append(result)

        # All results should be identical and use the same formatter
        assert all(result == "Thread-safe result" for result in results)
        assert mock_formatter.format_error.call_count == 10

    def test_error_message_edge_cases(self):
        """Test edge cases for error messages"""
        # Empty message
        result = ValidationErrorFormatter.format_error("")
        assert result == "❌ "

        result = ValidationErrorFormatter.format_error("", "Context")
        assert result == "❌ Context: "

        # Very long message
        long_message = "x" * 1000
        result = ValidationErrorFormatter.format_error(long_message)
        assert result == f"❌ {long_message}"

        # Special characters
        special_message = "Error with 'quotes' and \"double quotes\" and \n newlines"
        result = ValidationErrorFormatter.format_error(special_message, "Special context")
        assert result == f"❌ Special context: {special_message}"

    def test_integration_with_real_formatter(self):
        """Test integration with actual DecisionFormatter instance"""
        real_formatter = DecisionFormatter()
        ValidationErrorFormatter.initialize(real_formatter)

        result = ValidationErrorFormatter.format_error("Test integration", "Integration test")

        # Should produce formatted output (not just fallback)
        assert result != "❌ Integration test: Test integration"
        assert "Test integration" in result
        assert isinstance(result, str)
