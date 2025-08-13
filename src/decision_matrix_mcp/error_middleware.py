# MIT License
#
# Copyright (c) 2025 Democratize Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Error handling middleware for MCP tools."""

from collections.abc import Callable
import logging
from typing import Any, TypeVar

from .exceptions import DecisionMatrixError, ValidationError
from .services.response_service import ResponseService

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class MCPErrorHandler:
    """Centralized error handling middleware for MCP tools."""

    def __init__(self, response_service: ResponseService) -> None:
        """Initialize error handler with response service."""
        self.response_service = response_service
        self.logger = logging.getLogger(__name__)

    def handle_exception(self, exc: Exception, operation: str) -> dict[str, Any]:
        """Handle any exception and return standardized MCP error response.

        Args:
            exc: The exception that occurred
            operation: Name of the operation that failed

        Returns:
            Standardized error response dictionary
        """
        if isinstance(exc, DecisionMatrixError):
            return self._handle_decision_matrix_error(exc, operation)
        if isinstance(exc, ValueError):
            return self._handle_value_error(exc, operation)
        if isinstance(exc, KeyError):
            return self._handle_key_error(exc, operation)
        return self._handle_unexpected_error(exc, operation)

    def _handle_decision_matrix_error(
        self,
        exc: DecisionMatrixError,
        operation: str,
    ) -> dict[str, Any]:
        """Handle custom DecisionMatrixError exceptions."""
        self.logger.warning(
            "%s failed: %s",
            operation,
            exc,
            extra={
                "error_code": exc.error_code,
                "error_category": exc.error_category,
                "context": exc.context,
                "operation": operation,
            },
        )

        return self.response_service.create_error_response(
            message=exc.user_message,
            context=operation,
            error_code=exc.error_code,
            error_category=exc.error_category,
            recovery_suggestion=exc.recovery_suggestion,
            diagnostic_context=exc.context,
        )

    def _handle_validation_error(self, exc: ValidationError, operation: str) -> dict[str, Any]:
        """Handle validation errors specifically."""
        self.logger.warning("Validation error in %s: %s", operation, exc)

        return self.response_service.create_error_response(
            message=exc.user_message,
            context=operation,
            error_code=exc.error_code,
            error_category="CLIENT_ERROR",
            recovery_suggestion=exc.recovery_suggestion,
        )

    def _handle_value_error(self, exc: ValueError, operation: str) -> dict[str, Any]:
        """Handle ValueError exceptions."""
        self.logger.warning("Value error in %s: %s", operation, exc)

        return self.response_service.create_error_response(
            message="Invalid input provided",
            context=operation,
            error_code="DMX_1099",
            error_category="CLIENT_ERROR",
            recovery_suggestion="Check input values and try again",
            diagnostic_context={"original_error": str(exc)},
        )

    def _handle_key_error(self, exc: KeyError, operation: str) -> dict[str, Any]:
        """Handle KeyError exceptions."""
        self.logger.warning("Key error in %s: %s", operation, exc)

        return self.response_service.create_error_response(
            message="Required data missing",
            context=operation,
            error_code="DMX_1098",
            error_category="CLIENT_ERROR",
            recovery_suggestion="Ensure all required fields are provided",
            diagnostic_context={"missing_key": str(exc)},
        )

    def _handle_unexpected_error(self, exc: Exception, operation: str) -> dict[str, Any]:
        """Handle unexpected exceptions."""
        self.logger.exception("Unexpected error in %s", operation)

        return self.response_service.create_error_response(
            message="An unexpected error occurred",
            context=operation,
            error_code="DMX_9999",
            error_category="SERVER_ERROR",
            recovery_suggestion="Please try again or contact support if the issue persists",
            diagnostic_context={"error_type": type(exc).__name__, "error_message": str(exc)},
        )

    def with_error_handling(self, operation: str) -> Callable[[F], F]:
        """Decorator to add standardized error handling to MCP tool functions.

        Args:
            operation: Name of the operation for error context

        Returns:
            Decorator function
        """

        def decorator(func: F) -> F:
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self.handle_exception(e, operation)

            return wrapper

        return decorator
