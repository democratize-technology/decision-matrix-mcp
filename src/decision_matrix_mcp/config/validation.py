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

"""Configuration validation utilities for Decision Matrix MCP.

This module provides comprehensive validation for configuration values
including cross-field validation, security checks, and resource limit verification.
"""

from typing import Any

import psutil  # type: ignore[import-untyped]
from pydantic import ValidationError

from .schema import ConfigSchema


class ConfigValidationError(Exception):
    """Configuration validation error with detailed context."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class ConfigValidator:
    """Comprehensive configuration validator with security and resource checks."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.recommendations: list[str] = []

    def validate_config(self, config: ConfigSchema) -> None:
        """Perform comprehensive validation of configuration.

        Args:
            config: Configuration to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        self.warnings.clear()
        self.recommendations.clear()

        try:
            # Basic Pydantic validation
            config.model_dump()

            # Cross-field validation
            self._validate_cross_field_constraints(config)

            # Resource-based validation
            self._validate_system_resources(config)

            # Security validation
            self._validate_security_constraints(config)

            # Performance validation
            self._validate_performance_constraints(config)

        except ValidationError as e:
            error_dicts = []
            for error in e.errors():
                if hasattr(error, "dict"):
                    error_dicts.append(error.dict())  # type: ignore[attr-defined]
                else:
                    error_dicts.append({"error": str(error)})
            raise ConfigValidationError("Configuration validation failed", error_dicts) from e
        except Exception as e:
            msg = f"Unexpected validation error: {e}"
            raise ConfigValidationError(msg) from e

    def _validate_cross_field_constraints(self, config: ConfigSchema) -> None:
        """Validate constraints that span multiple configuration fields."""
        # Session constraints
        if config.session.lru_eviction_batch_size >= config.session.max_active_sessions:
            raise ConfigValidationError(
                f"LRU eviction batch size ({config.session.lru_eviction_batch_size}) "
                f"must be less than max active sessions ({config.session.max_active_sessions})",
            )

        if config.session.default_max_sessions > config.session.max_active_sessions:
            self.warnings.append(
                f"Default max sessions ({config.session.default_max_sessions}) "
                f"exceeds max active sessions ({config.session.max_active_sessions}). "
                f"This may cause unexpected LRU eviction behavior.",
            )

        # Validation constraints
        if config.validation.max_options_allowed <= config.validation.min_options_required:
            raise ConfigValidationError(
                f"Max options allowed ({config.validation.max_options_allowed}) "
                f"must be greater than min options required ({config.validation.min_options_required})",
            )

        if config.validation.max_criterion_weight <= config.validation.min_criterion_weight:
            raise ConfigValidationError(
                f"Max criterion weight ({config.validation.max_criterion_weight}) "
                f"must be greater than min criterion weight ({config.validation.min_criterion_weight})",
            )

        # Performance constraints
        if config.performance.cot_summary_timeout_seconds >= config.performance.cot_timeout_seconds:
            self.warnings.append(
                f"CoT summary timeout ({config.performance.cot_summary_timeout_seconds}s) "
                f"should be less than CoT timeout ({config.performance.cot_timeout_seconds}s)",
            )

    def _validate_system_resources(self, config: ConfigSchema) -> None:
        """Validate configuration against available system resources."""
        try:
            # Check memory constraints
            available_memory_gb = psutil.virtual_memory().total / (1024**3)

            # Rough memory estimation:
            # - Each session: ~50KB (conservative estimate)
            # - Concurrent evaluations: ~10MB each (LLM context)
            estimated_session_memory_mb = (config.session.max_active_sessions * 50) / 1024
            estimated_concurrent_memory_mb = config.performance.max_concurrent_evaluations * 10
            total_estimated_mb = estimated_session_memory_mb + estimated_concurrent_memory_mb

            if total_estimated_mb > (available_memory_gb * 1024 * 0.8):  # 80% of available memory
                self.warnings.append(
                    f"Configuration may use ~{total_estimated_mb:.1f}MB memory "
                    f"(available: {available_memory_gb:.1f}GB). Consider reducing limits.",
                )

            # Check CPU constraints
            cpu_count = psutil.cpu_count()
            if config.performance.max_concurrent_evaluations > cpu_count * 2:
                self.recommendations.append(
                    f"Max concurrent evaluations ({config.performance.max_concurrent_evaluations}) "
                    f"exceeds 2x CPU cores ({cpu_count}). Consider reducing for optimal performance.",
                )

        except (OSError, ImportError, AttributeError) as e:
            # System resource checks are best-effort
            self.warnings.append(f"Could not validate system resources: {e}")

    def _validate_security_constraints(self, config: ConfigSchema) -> None:
        """Validate security-related configuration constraints."""
        # Check for excessively large limits that could enable DoS
        if config.validation.max_description_length > 10000:
            self.warnings.append(
                f"Very large description limit ({config.validation.max_description_length}) "
                f"could enable DoS attacks. Consider reducing.",
            )

        if config.validation.max_options_allowed > 50:
            self.warnings.append(
                f"Very large option limit ({config.validation.max_options_allowed}) "
                f"could cause performance issues. Consider reducing.",
            )

        if config.session.max_active_sessions > 1000:
            self.warnings.append(
                f"Very large session limit ({config.session.max_active_sessions}) "
                f"could consume excessive memory. Consider reducing.",
            )

        # Check timeout configurations
        if config.performance.request_timeout_seconds > 300:  # 5 minutes
            self.warnings.append(
                f"Very long request timeout ({config.performance.request_timeout_seconds}s) "
                f"could lead to resource exhaustion. Consider reducing.",
            )

        # Validate environment-specific settings
        if config.environment == "production" and config.debug_mode:
            self.warnings.append(
                "Debug mode is enabled in production environment. "
                "This may expose sensitive information.",
            )

    def _validate_performance_constraints(self, config: ConfigSchema) -> None:
        """Validate performance-related configuration constraints."""
        # Check retry configuration
        if config.performance.max_retries > 5:
            self.recommendations.append(
                f"High retry count ({config.performance.max_retries}) "
                f"may cause long delays on persistent failures.",
            )

        if config.performance.retry_delay_seconds > 10:
            self.recommendations.append(
                f"Long retry delay ({config.performance.retry_delay_seconds}s) "
                f"may cause poor user experience.",
            )

        # Check concurrency vs timeout balance
        total_potential_timeout = (
            config.performance.max_concurrent_evaluations
            * config.performance.request_timeout_seconds
        )
        if total_potential_timeout > 600:  # 10 minutes
            self.recommendations.append(
                f"High concurrency + timeout combination could cause "
                f"very long evaluation times ({total_potential_timeout:.1f}s max). "
                f"Consider reducing concurrent evaluations or timeouts.",
            )

        # Check session cleanup frequency
        cleanup_ratio = config.session.default_cleanup_interval_minutes / (
            config.session.default_session_ttl_hours * 60
        )
        if cleanup_ratio > 0.5:  # Cleanup more than half as often as TTL
            self.recommendations.append(
                "Frequent cleanup interval may cause unnecessary overhead. "
                "Consider increasing cleanup interval relative to session TTL.",
            )

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of validation results including warnings and recommendations."""
        return {
            "status": "valid",
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "warning_count": len(self.warnings),
            "recommendation_count": len(self.recommendations),
        }
