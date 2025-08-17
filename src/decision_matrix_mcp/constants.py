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

"""Configuration-based constants for the decision matrix MCP server.

This module provides backward-compatible constants that now use the
ConfigManager for dynamic configuration via environment variables.

MIGRATION NOTE: This module maintains backward compatibility with existing
code while transitioning to the new configuration system. All constants
now dynamically read from the ConfigManager instead of hardcoded values.

Environment Variables:
- All configuration can be overridden using DMM_* environment variables
- See config.defaults.ENV_VAR_MAPPING for the complete list
- Example: DMM_MAX_OPTIONS_ALLOWED=30 to change the max options limit

For new code, prefer importing directly from the config module:
    from decision_matrix_mcp.config import config
    max_options = config.validation.max_options_allowed
"""

from .config import config


class ValidationLimits:
    """Validation limits for input sanitization and business logic.

    These properties dynamically read from the ConfigManager, allowing
    runtime configuration via environment variables.
    """

    @property
    def MAX_SESSION_ID_LENGTH(self) -> int:  # noqa: N802
        """Maximum length for session IDs."""
        return config.validation.max_session_id_length

    @property
    def MAX_TOPIC_LENGTH(self) -> int:  # noqa: N802
        """Maximum length for decision topics."""
        return config.validation.max_topic_length

    @property
    def MAX_OPTION_NAME_LENGTH(self) -> int:  # noqa: N802
        """Maximum length for option names."""
        return config.validation.max_option_name_length

    @property
    def MAX_CRITERION_NAME_LENGTH(self) -> int:  # noqa: N802
        """Maximum length for criterion names."""
        return config.validation.max_criterion_name_length

    @property
    def MAX_DESCRIPTION_LENGTH(self) -> int:  # noqa: N802
        """Maximum length for descriptions."""
        return config.validation.max_description_length

    @property
    def MIN_OPTIONS_REQUIRED(self) -> int:  # noqa: N802
        """Minimum number of options required for decision analysis."""
        return config.validation.min_options_required

    @property
    def MAX_OPTIONS_ALLOWED(self) -> int:  # noqa: N802
        """Maximum number of options allowed per decision."""
        return config.validation.max_options_allowed

    @property
    def MAX_CRITERIA_ALLOWED(self) -> int:  # noqa: N802
        """Maximum number of criteria allowed per decision."""
        return config.validation.max_criteria_allowed

    @property
    def MIN_CRITERION_WEIGHT(self) -> float:  # noqa: N802
        """Minimum weight value for criteria."""
        return config.validation.min_criterion_weight

    @property
    def MAX_CRITERION_WEIGHT(self) -> float:  # noqa: N802
        """Maximum weight value for criteria."""
        return config.validation.max_criterion_weight


class SessionLimits:
    """Session management limits for memory safety and resource control.

    These properties dynamically read from the ConfigManager, allowing
    runtime configuration via environment variables.
    """

    @property
    def MAX_ACTIVE_SESSIONS(self) -> int:  # noqa: N802
        """Maximum active sessions before LRU eviction kicks in."""
        return config.session.max_active_sessions

    @property
    def LRU_EVICTION_BATCH_SIZE(self) -> int:  # noqa: N802
        """Number of sessions to evict when reaching limit (batch eviction)."""
        return config.session.lru_eviction_batch_size

    @property
    def DEFAULT_MAX_SESSIONS(self) -> int:  # noqa: N802
        """Default maximum sessions per SessionManager instance."""
        return config.session.default_max_sessions

    @property
    def DEFAULT_SESSION_TTL_HOURS(self) -> int:  # noqa: N802
        """Default session time-to-live in hours."""
        return config.session.default_session_ttl_hours

    @property
    def DEFAULT_CLEANUP_INTERVAL_MINUTES(self) -> int:  # noqa: N802
        """Default cleanup interval in minutes."""
        return config.session.default_cleanup_interval_minutes


ValidationLimits = ValidationLimits()
SessionLimits = SessionLimits()
