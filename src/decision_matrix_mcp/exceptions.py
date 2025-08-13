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

"""Custom exceptions for decision matrix MCP server."""

from datetime import datetime, timezone
from typing import Any


class DecisionMatrixError(Exception):
    """Base exception for all decision matrix errors."""

    ERROR_CATEGORY = "GENERAL"
    ERROR_CODE = "DMX_0000"

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        recovery_suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or "An error occurred"
        self.error_code = error_code or self.ERROR_CODE
        self.error_category = self.ERROR_CATEGORY
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = datetime.now(timezone.utc)


class SessionError(DecisionMatrixError):
    """Errors related to session management."""

    ERROR_CATEGORY = "CLIENT_ERROR"
    ERROR_CODE = "DMX_2000"


class ValidationError(DecisionMatrixError):
    """Input validation errors."""

    ERROR_CATEGORY = "CLIENT_ERROR"
    ERROR_CODE = "DMX_1000"


class LLMBackendError(DecisionMatrixError):
    """Errors from LLM backend calls."""

    ERROR_CATEGORY = "SERVER_ERROR"
    ERROR_CODE = "DMX_3000"

    def __init__(
        self,
        backend: str,
        message: str,
        user_message: str | None = None,
        original_error: Exception | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        recovery_suggestion: str | None = None,
    ) -> None:
        context = context or {}
        context["backend"] = backend
        if original_error:
            context["original_error"] = str(original_error)
        super().__init__(message, user_message, error_code, context, recovery_suggestion)
        self.backend = backend
        self.original_error = original_error


class LLMConfigurationError(LLMBackendError):
    """Missing dependencies or configuration for LLM backend."""

    ERROR_CODE = "DMX_3001"

    def __init__(
        self,
        backend: str,
        message: str,
        user_message: str | None = None,
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        user_message = user_message or f"{backend} backend not properly configured"
        recovery_suggestion = (
            f"Ensure {backend} backend is properly configured with required dependencies"
        )
        super().__init__(
            backend,
            message,
            user_message,
            original_error,
            self.ERROR_CODE,
            context,
            recovery_suggestion,
        )


class LLMAPIError(LLMBackendError):
    """API call failures for LLM backend."""

    ERROR_CODE = "DMX_3002"

    def __init__(
        self,
        backend: str,
        message: str,
        user_message: str | None = None,
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        user_message = user_message or "LLM service temporarily unavailable"
        recovery_suggestion = "Retry the operation or check backend service status"
        super().__init__(
            backend,
            message,
            user_message,
            original_error,
            self.ERROR_CODE,
            context,
            recovery_suggestion,
        )


class ConfigurationError(DecisionMatrixError):
    """Configuration or setup errors."""

    ERROR_CATEGORY = "SERVER_ERROR"
    ERROR_CODE = "DMX_4000"


class ResourceLimitError(DecisionMatrixError):
    """Resource limit exceeded errors."""

    ERROR_CATEGORY = "CLIENT_ERROR"
    ERROR_CODE = "DMX_5000"


class ChainOfThoughtError(DecisionMatrixError):
    """Errors specific to Chain of Thought reasoning."""

    ERROR_CATEGORY = "SERVER_ERROR"
    ERROR_CODE = "DMX_6000"


class CoTTimeoutError(ChainOfThoughtError):
    """Chain of Thought processing timeout."""

    ERROR_CODE = "DMX_6001"

    def __init__(self, timeout: float, message: str | None = None) -> None:
        message = message or f"Chain of Thought evaluation timed out after {timeout} seconds"
        user_message = "Reasoning process took too long and was terminated"
        context = {"timeout_seconds": timeout}
        recovery_suggestion = "Try reducing the complexity of the evaluation or increase timeout"
        super().__init__(message, user_message, self.ERROR_CODE, context, recovery_suggestion)
        self.timeout = timeout


class CoTProcessingError(ChainOfThoughtError):
    """Error during Chain of Thought processing."""

    ERROR_CODE = "DMX_6002"

    def __init__(self, message: str, stage: str | None = None) -> None:
        user_message = "Error occurred during structured reasoning"
        context = {"stage": stage} if stage else {}
        recovery_suggestion = "Check evaluation criteria and retry"
        super().__init__(message, user_message, self.ERROR_CODE, context, recovery_suggestion)
        self.stage = stage


class InputSanitizationError(ValidationError):
    """Input was rejected during sanitization."""

    ERROR_CODE = "DMX_1001"

    def __init__(self, field: str, reason: str) -> None:
        message = f"Input sanitization failed for {field}: {reason}"
        user_message = f"Invalid input detected in {field}"
        context = {"field": field, "reason": reason}
        recovery_suggestion = f"Provide valid input for {field}"
        super().__init__(message, user_message, self.ERROR_CODE, context, recovery_suggestion)
        self.field = field
        self.reason = reason
