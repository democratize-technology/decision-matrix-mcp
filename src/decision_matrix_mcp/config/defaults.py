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

"""Default configuration values for Decision Matrix MCP.

This module provides the default configuration that ensures the system
works out-of-the-box without any environment variable configuration.
"""

from typing import Any

from .schema import ConfigSchema

# Default configuration instance
DEFAULT_CONFIG = ConfigSchema()

# Environment variable mapping for easy reference
ENV_VAR_MAPPING = {
    # Validation limits
    "DMM_MAX_SESSION_ID_LENGTH": "validation.max_session_id_length",
    "DMM_MAX_TOPIC_LENGTH": "validation.max_topic_length",
    "DMM_MAX_OPTION_NAME_LENGTH": "validation.max_option_name_length",
    "DMM_MAX_CRITERION_NAME_LENGTH": "validation.max_criterion_name_length",
    "DMM_MAX_DESCRIPTION_LENGTH": "validation.max_description_length",
    "DMM_MIN_OPTIONS_REQUIRED": "validation.min_options_required",
    "DMM_MAX_OPTIONS_ALLOWED": "validation.max_options_allowed",
    "DMM_MAX_CRITERIA_ALLOWED": "validation.max_criteria_allowed",
    "DMM_MIN_CRITERION_WEIGHT": "validation.min_criterion_weight",
    "DMM_MAX_CRITERION_WEIGHT": "validation.max_criterion_weight",
    # Session limits
    "DMM_MAX_ACTIVE_SESSIONS": "session.max_active_sessions",
    "DMM_LRU_EVICTION_BATCH_SIZE": "session.lru_eviction_batch_size",
    "DMM_DEFAULT_MAX_SESSIONS": "session.default_max_sessions",
    "DMM_DEFAULT_SESSION_TTL_HOURS": "session.default_session_ttl_hours",
    "DMM_DEFAULT_CLEANUP_INTERVAL_MINUTES": "session.default_cleanup_interval_minutes",
    # Performance limits
    "DMM_MAX_RETRIES": "performance.max_retries",
    "DMM_RETRY_DELAY_SECONDS": "performance.retry_delay_seconds",
    "DMM_MAX_CONCURRENT_EVALUATIONS": "performance.max_concurrent_evaluations",
    "DMM_REQUEST_TIMEOUT_SECONDS": "performance.request_timeout_seconds",
    "DMM_COT_TIMEOUT_SECONDS": "performance.cot_timeout_seconds",
    "DMM_COT_SUMMARY_TIMEOUT_SECONDS": "performance.cot_summary_timeout_seconds",
    # Backend configuration
    "DMM_BEDROCK_MODEL": "backend.bedrock_model",
    "DMM_LITELLM_MODEL": "backend.litellm_model",
    "DMM_OLLAMA_MODEL": "backend.ollama_model",
    "DMM_BEDROCK_TIMEOUT_SECONDS": "backend.bedrock_timeout_seconds",
    "DMM_LITELLM_TIMEOUT_SECONDS": "backend.litellm_timeout_seconds",
    "DMM_OLLAMA_TIMEOUT_SECONDS": "backend.ollama_timeout_seconds",
    "DMM_DEFAULT_TEMPERATURE": "backend.default_temperature",
    # Environment settings
    "DMM_ENVIRONMENT": "environment",
    "DMM_DEBUG_MODE": "debug_mode",
    "DMM_CONFIG_VERSION": "config_version",
}

# Type mapping for environment variable conversion
ENV_VAR_TYPES = {
    # Integer types
    "DMM_MAX_SESSION_ID_LENGTH": int,
    "DMM_MAX_TOPIC_LENGTH": int,
    "DMM_MAX_OPTION_NAME_LENGTH": int,
    "DMM_MAX_CRITERION_NAME_LENGTH": int,
    "DMM_MAX_DESCRIPTION_LENGTH": int,
    "DMM_MIN_OPTIONS_REQUIRED": int,
    "DMM_MAX_OPTIONS_ALLOWED": int,
    "DMM_MAX_CRITERIA_ALLOWED": int,
    "DMM_MAX_ACTIVE_SESSIONS": int,
    "DMM_LRU_EVICTION_BATCH_SIZE": int,
    "DMM_DEFAULT_MAX_SESSIONS": int,
    "DMM_DEFAULT_SESSION_TTL_HOURS": int,
    "DMM_DEFAULT_CLEANUP_INTERVAL_MINUTES": int,
    "DMM_MAX_RETRIES": int,
    "DMM_MAX_CONCURRENT_EVALUATIONS": int,
    # Float types
    "DMM_MIN_CRITERION_WEIGHT": float,
    "DMM_MAX_CRITERION_WEIGHT": float,
    "DMM_RETRY_DELAY_SECONDS": float,
    "DMM_REQUEST_TIMEOUT_SECONDS": float,
    "DMM_COT_TIMEOUT_SECONDS": float,
    "DMM_COT_SUMMARY_TIMEOUT_SECONDS": float,
    "DMM_BEDROCK_TIMEOUT_SECONDS": float,
    "DMM_LITELLM_TIMEOUT_SECONDS": float,
    "DMM_OLLAMA_TIMEOUT_SECONDS": float,
    "DMM_DEFAULT_TEMPERATURE": float,
    # Boolean types
    "DMM_DEBUG_MODE": bool,
    # String types (default)
    "DMM_BEDROCK_MODEL": str,
    "DMM_LITELLM_MODEL": str,
    "DMM_OLLAMA_MODEL": str,
    "DMM_ENVIRONMENT": str,
    "DMM_CONFIG_VERSION": str,
}

# Configuration profiles for different environments
ENVIRONMENT_PROFILES: dict[str, dict[str, Any]] = {
    "development": {
        "debug_mode": True,
        "validation.max_options_allowed": 10,  # Smaller limits for dev
        "session.max_active_sessions": 20,
        "session.default_max_sessions": 15,  # Must be less than max_active_sessions
        "performance.request_timeout_seconds": 60.0,  # Longer timeouts for debugging
        "performance.cot_timeout_seconds": 60.0,
    },
    "staging": {
        "debug_mode": False,
        "validation.max_options_allowed": 15,
        "session.max_active_sessions": 50,
        "session.default_max_sessions": 40,  # Must be less than max_active_sessions
        "performance.request_timeout_seconds": 45.0,
        "performance.cot_timeout_seconds": 45.0,
    },
    "production": {
        "debug_mode": False,
        # Use all default values for production
    },
}


def get_profile_overrides(environment: str) -> dict[str, Any]:
    """Get configuration overrides for a specific environment profile."""
    return ENVIRONMENT_PROFILES.get(environment, {})
