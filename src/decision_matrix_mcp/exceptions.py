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
