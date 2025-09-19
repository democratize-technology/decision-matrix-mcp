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

"""Configuration manager for Decision Matrix MCP.

This module provides the central ConfigManager class that handles:
- Environment variable loading with type conversion
- Configuration validation and cross-field checks
- Environment profile application
- Thread-safe configuration access
- Runtime configuration updates
"""

import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

from .defaults import DEFAULT_CONFIG, ENV_VAR_MAPPING, ENV_VAR_TYPES, get_profile_overrides
from .schema import ConfigSchema, SessionLimitsConfig, ValidationLimitsConfig
from .validation import ConfigValidationError, ConfigValidator

logger = logging.getLogger(__name__)


class ConfigManager:
    """Thread-safe configuration manager with environment variable support.

    Features:
    - Environment variable loading with DMM_ prefix
    - Type conversion and validation
    - Environment profiles (development, staging, production)
    - Configuration file support (JSON/YAML)
    - Runtime configuration updates
    - Comprehensive validation with warnings and recommendations
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConfigManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration manager."""
        if hasattr(self, "_initialized"):
            return

        self._config_lock = threading.RLock()
        self._config: ConfigSchema = DEFAULT_CONFIG.model_copy(deep=True)
        self._validator = ConfigValidator()
        self._loaded_from_env = False
        self._loaded_from_file = False
        self._initialization_errors: list[str] = []

        # Load configuration on first initialization
        try:
            self._load_configuration()
        except Exception as e:
            logger.exception("Failed to load configuration")
            self._initialization_errors.append(str(e))

        self._initialized = True

    def _load_configuration(self) -> None:
        """Load configuration from environment variables and files."""
        with self._config_lock:
            # Start with default configuration
            config_data = self._config.model_dump()

            # Load from configuration files first
            self._load_from_files(config_data)

            # Override with environment variables
            self._load_from_environment(config_data)

            # Apply environment profile overrides
            self._apply_environment_profile(config_data)

            # Create and validate new configuration
            try:
                new_config = ConfigSchema(**config_data)
                self._validator.validate_config(new_config)
                self._config = new_config

                logger.info("Configuration loaded successfully")
                if self._validator.warnings:
                    logger.warning("Configuration warnings: %s", self._validator.warnings)
                if self._validator.recommendations:
                    logger.info(
                        "Configuration recommendations: %s",
                        self._validator.recommendations,
                    )

            except Exception as e:
                logger.exception("Configuration validation failed")
                raise ConfigValidationError(f"Invalid configuration: {e}") from e

    def _load_from_files(self, config_data: dict[str, Any]) -> None:
        """Load configuration from JSON/YAML files."""
        config_paths = [
            Path("/etc/decision-matrix-mcp/config.json"),
            Path("/etc/decision-matrix-mcp/config.yaml"),
            Path("/etc/decision-matrix-mcp/config.yml"),
            Path.home() / ".config" / "decision-matrix-mcp" / "config.json",
            Path.home() / ".config" / "decision-matrix-mcp" / "config.yaml",
            Path.home() / ".config" / "decision-matrix-mcp" / "config.yml",
            Path("config.json"),
            Path("config.yaml"),
            Path("config.yml"),
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with config_path.open() as f:
                        if config_path.suffix in [".yaml", ".yml"]:
                            if yaml is None:
                                logger.warning(
                                    "YAML support not available, skipping %s",
                                    config_path,
                                )
                                continue
                            file_config = yaml.safe_load(f)
                        else:
                            file_config = json.load(f)

                    if file_config:
                        self._merge_config(config_data, file_config)
                        self._loaded_from_file = True
                        logger.info("Loaded configuration from %s", config_path)
                        break

                except (FileNotFoundError, json.JSONDecodeError, PermissionError, OSError) as e:
                    logger.warning("Failed to load config from %s: %s", config_path, e)

    def _load_from_environment(self, config_data: dict[str, Any]) -> None:
        """Load configuration from environment variables with DMM_ prefix."""
        env_vars_found = []

        for env_var, config_path in ENV_VAR_MAPPING.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert to appropriate type
                    var_type = ENV_VAR_TYPES.get(env_var, str)
                    if var_type is bool:
                        # Handle boolean environment variables
                        converted_value = env_value.lower() in ("true", "1", "yes", "on")
                    else:
                        converted_value = var_type(env_value)

                    # Set nested configuration value
                    self._set_nested_value(config_data, config_path, converted_value)
                    env_vars_found.append(env_var)

                except (ValueError, TypeError) as e:
                    logger.warning("Invalid value for %s='%s': %s", env_var, env_value, e)

        if env_vars_found:
            self._loaded_from_env = True
            logger.info(
                "Loaded %d configuration values from environment variables",
                len(env_vars_found),
            )

    def _apply_environment_profile(self, config_data: dict[str, Any]) -> None:
        """Apply environment-specific configuration overrides."""
        environment = config_data.get("environment", "production")
        profile_overrides = get_profile_overrides(environment)

        if profile_overrides:
            for config_path, value in profile_overrides.items():
                self._set_nested_value(config_data, config_path, value)
            logger.info("Applied %s environment profile", environment)

    def _set_nested_value(self, data: dict[str, Any], path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation path."""
        keys = path.split(".")
        current = data

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _merge_config(self, base_config: dict[str, Any], new_config: dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in new_config.items():
            if (
                key in base_config
                and isinstance(base_config[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value

    @property
    def config(self) -> ConfigSchema:
        """Get current configuration (thread-safe)."""
        with self._config_lock:
            result: ConfigSchema = self._config.model_copy(deep=True)
            return result

    @property
    def validation(self) -> ValidationLimitsConfig:
        """Get validation limits configuration."""
        return self._config.validation

    @property
    def session(self) -> SessionLimitsConfig:
        """Get session limits configuration."""
        return self._config.session

    @property
    def performance(self) -> Any:
        """Get performance limits configuration."""
        return self._config.performance

    @property
    def backend(self) -> Any:
        """Get backend configuration."""
        return self._config.backend

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration at runtime (thread-safe).

        Args:
            **kwargs: Configuration values to update using dot notation

        Example:
            config.update_config(
                validation__max_options_allowed=30,
                performance__request_timeout_seconds=45.0
            )
        """
        with self._config_lock:
            config_data = self._config.model_dump()

            for key, value in kwargs.items():
                # Convert double underscore to dot notation
                config_path = key.replace("__", ".")
                self._set_nested_value(config_data, config_path, value)

            # Validate and apply new configuration
            try:
                new_config = ConfigSchema(**config_data)
                self._validator.validate_config(new_config)
                self._config = new_config
                logger.info("Configuration updated: %s", list(kwargs.keys()))

            except Exception as e:
                logger.exception("Configuration update failed")
                raise ConfigValidationError(f"Invalid configuration update: {e}") from e

    def reload_configuration(self) -> None:
        """Reload configuration from environment and files."""
        logger.info("Reloading configuration...")
        self._loaded_from_env = False
        self._loaded_from_file = False
        self._initialization_errors.clear()
        self._load_configuration()

    def get_config_summary(self) -> dict[str, Any]:
        """Get comprehensive configuration summary."""
        validation_summary = self._validator.get_validation_summary()

        return {
            "config_version": self._config.config_version,
            "environment": self._config.environment,
            "debug_mode": self._config.debug_mode,
            "loaded_from_env": self._loaded_from_env,
            "loaded_from_file": self._loaded_from_file,
            "initialization_errors": self._initialization_errors,
            "validation": validation_summary,
            "limits_summary": {
                "max_active_sessions": self._config.session.max_active_sessions,
                "max_options_allowed": self._config.validation.max_options_allowed,
                "max_concurrent_evaluations": self._config.performance.max_concurrent_evaluations,
                "request_timeout_seconds": self._config.performance.request_timeout_seconds,
            },
        }

    def export_config(self, format: str = "json") -> str:
        """Export current configuration to JSON or YAML format."""
        config_dict = self._config.model_dump()

        if format.lower() == "yaml":
            if yaml is None:
                raise ValueError("YAML support not available. Install PyYAML: pip install pyyaml")
            return str(yaml.dump(config_dict, default_flow_style=False, indent=2))
        return json.dumps(config_dict, indent=2)

    def get_env_var_help(self) -> dict[str, str]:
        """Get help text for all supported environment variables."""
        help_text = {}

        for env_var, config_path in ENV_VAR_MAPPING.items():
            var_type = ENV_VAR_TYPES.get(env_var, str)
            help_text[env_var] = f"Type: {var_type.__name__}, Path: {config_path}"

        return help_text


# Global configuration instance
config = ConfigManager()
