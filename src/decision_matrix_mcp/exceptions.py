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

"""Custom exceptions for decision matrix MCP server"""


class DecisionMatrixError(Exception):
    """Base exception for all decision matrix errors"""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or "An error occurred"


class SessionError(DecisionMatrixError):
    """Errors related to session management"""

    pass


class ValidationError(DecisionMatrixError):
    """Input validation errors"""

    pass


class LLMBackendError(DecisionMatrixError):
    """Errors from LLM backend calls"""

    def __init__(
        self,
        backend: str,
        message: str,
        user_message: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, user_message)
        self.backend = backend
        self.original_error = original_error


class LLMConfigurationError(LLMBackendError):
    """Missing dependencies or configuration for LLM backend"""

    def __init__(
        self,
        backend: str,
        message: str,
        user_message: str | None = None,
        original_error: Exception | None = None,
    ):
        user_message = user_message or f"{backend} backend not properly configured"
        super().__init__(backend, message, user_message, original_error)


class LLMAPIError(LLMBackendError):
    """API call failures for LLM backend"""

    def __init__(
        self,
        backend: str,
        message: str,
        user_message: str | None = None,
        original_error: Exception | None = None,
    ):
        user_message = user_message or "LLM service temporarily unavailable"
        super().__init__(backend, message, user_message, original_error)


class ConfigurationError(DecisionMatrixError):
    """Configuration or setup errors"""

    pass


class ResourceLimitError(DecisionMatrixError):
    """Resource limit exceeded errors"""

    pass


class ChainOfThoughtError(DecisionMatrixError):
    """Errors specific to Chain of Thought reasoning"""

    pass


class CoTTimeoutError(ChainOfThoughtError):
    """Chain of Thought processing timeout"""

    def __init__(self, timeout: float, message: str | None = None):
        message = message or f"Chain of Thought evaluation timed out after {timeout} seconds"
        super().__init__(message, "Reasoning process took too long and was terminated")
        self.timeout = timeout


class CoTProcessingError(ChainOfThoughtError):
    """Error during Chain of Thought processing"""

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message, "Error occurred during structured reasoning")
        self.stage = stage


class InputSanitizationError(ValidationError):
    """Input was rejected during sanitization"""

    def __init__(self, field: str, reason: str):
        message = f"Input sanitization failed for {field}: {reason}"
        super().__init__(message, f"Invalid input detected in {field}")
        self.field = field
        self.reason = reason
