"""
Validation constants for the decision matrix MCP server.

This module centralizes all validation limits and constants to ensure consistency
across the codebase and make configuration changes easier.
"""


class ValidationLimits:
    """Validation limits for input sanitization and business logic."""

    # Session validation
    MAX_SESSION_ID_LENGTH = 100

    # Content validation
    MAX_TOPIC_LENGTH = 500
    MAX_OPTION_NAME_LENGTH = 200
    MAX_CRITERION_NAME_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 1000

    # Business logic limits
    MIN_OPTIONS_REQUIRED = 2
    MAX_OPTIONS_ALLOWED = 20

    # Weight validation
    MIN_CRITERION_WEIGHT = 0.1
    MAX_CRITERION_WEIGHT = 10.0
