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

    pass


class ConfigurationError(DecisionMatrixError):
    """Configuration or setup errors"""

    pass


class ResourceLimitError(DecisionMatrixError):
    """Resource limit exceeded errors"""

    pass
