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

"""
Comprehensive tests for the configuration management system.

Tests cover:
- Environment variable loading and validation
- Configuration schema validation
- Cross-field constraint validation
- Resource-based validation
- Security constraint validation
- Backward compatibility with constants
- Configuration file loading
- Runtime configuration updates
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from decision_matrix_mcp.config import (
    DEFAULT_CONFIG,
    ConfigManager,
    ConfigSchema,
    ConfigValidationError,
    ConfigValidator,
)

# Import constants for backward compatibility testing
from decision_matrix_mcp.constants import SessionLimits, ValidationLimits


class TestConfigSchema:
    """Test configuration schema validation and constraints."""

    def test_default_config_is_valid(self):
        """Test that the default configuration is valid."""
        config = ConfigSchema()
        validator = ConfigValidator()

        # Should not raise any validation errors
        validator.validate_config(config)

        # Check some default values
        assert config.validation.max_options_allowed == 20
        assert config.session.max_active_sessions == 100
        assert config.performance.max_retries == 3
        assert config.backend.bedrock_model == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_validation_limits_constraints(self):
        """Test validation limit constraints."""
        # Test valid configuration
        config = ConfigSchema()
        config.validation.min_options_required = 2
        config.validation.max_options_allowed = 10
        config.validation.min_criterion_weight = 0.1
        config.validation.max_criterion_weight = 5.0

        validator = ConfigValidator()
        validator.validate_config(config)  # Should not raise

        # Test invalid configuration - max <= min
        with pytest.raises(Exception):  # Pydantic validation error
            ConfigSchema(
                validation={
                    "min_options_required": 5,
                    "max_options_allowed": 3,  # Invalid: less than min
                },
            )

    def test_session_limits_constraints(self):
        """Test session limit constraints."""
        # Test valid configuration
        config = ConfigSchema()
        config.session.max_active_sessions = 100
        config.session.lru_eviction_batch_size = 10
        config.session.default_max_sessions = 50

        validator = ConfigValidator()
        validator.validate_config(config)  # Should not raise

        # Test invalid configuration - batch size >= max sessions
        with pytest.raises(ConfigValidationError):
            invalid_config = ConfigSchema()
            invalid_config.session.max_active_sessions = 10
            invalid_config.session.lru_eviction_batch_size = 15  # Invalid: >= max sessions
            validator.validate_config(invalid_config)

    def test_performance_limits_validation(self):
        """Test performance limit validation."""
        config = ConfigSchema()
        config.performance.max_retries = 5
        config.performance.retry_delay_seconds = 2.0
        config.performance.request_timeout_seconds = 30.0
        config.performance.cot_timeout_seconds = 45.0
        config.performance.cot_summary_timeout_seconds = 5.0

        validator = ConfigValidator()
        validator.validate_config(config)

        # Should generate warning for long timeout
        config.performance.request_timeout_seconds = 400.0  # Very long timeout
        validator.validate_config(config)
        assert len(validator.warnings) > 0
        assert any("long request timeout" in w.lower() for w in validator.warnings)

    def test_backend_configuration(self):
        """Test backend configuration validation."""
        config = ConfigSchema()
        config.backend.bedrock_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        config.backend.litellm_model = "gpt-4o-mini"
        config.backend.ollama_model = "llama3.1:8b"
        config.backend.default_temperature = 0.1

        validator = ConfigValidator()
        validator.validate_config(config)  # Should not raise


class TestEnvironmentVariableLoading:
    """Test environment variable loading and type conversion."""

    def test_integer_env_var_conversion(self):
        """Test integer environment variable conversion."""
        with patch.dict(
            os.environ,
            {
                "DMM_MAX_OPTIONS_ALLOWED": "30",
                "DMM_MAX_ACTIVE_SESSIONS": "200",
            },
        ):
            # Create new config manager to pick up env vars
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.validation.max_options_allowed == 30
            assert config_manager.config.session.max_active_sessions == 200

    def test_float_env_var_conversion(self):
        """Test float environment variable conversion."""
        with patch.dict(
            os.environ,
            {
                "DMM_MIN_CRITERION_WEIGHT": "0.5",
                "DMM_REQUEST_TIMEOUT_SECONDS": "45.5",
            },
        ):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.validation.min_criterion_weight == 0.5
            assert config_manager.config.performance.request_timeout_seconds == 45.5

    def test_boolean_env_var_conversion(self):
        """Test boolean environment variable conversion."""
        with patch.dict(
            os.environ,
            {
                "DMM_DEBUG_MODE": "true",
            },
        ):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.debug_mode is True

        with patch.dict(
            os.environ,
            {
                "DMM_DEBUG_MODE": "false",
            },
        ):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.debug_mode is False

    def test_string_env_var_conversion(self):
        """Test string environment variable conversion."""
        with patch.dict(
            os.environ,
            {
                "BEDROCK_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
                "DMM_ENVIRONMENT": "development",
            },
        ):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert (
                config_manager.config.backend.bedrock_model
                == "anthropic.claude-3-haiku-20240307-v1:0"
            )
            assert config_manager.config.environment == "development"

    def test_invalid_env_var_value(self):
        """Test handling of invalid environment variable values."""
        # Reset singleton state for test isolation
        ConfigManager._instance = None

        with patch.dict(
            os.environ,
            {
                "DMM_MAX_OPTIONS_ALLOWED": "invalid_integer",
            },
        ):
            # Should not crash, should use default value
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            # Should fall back to default
            assert (
                config_manager.config.validation.max_options_allowed
                == DEFAULT_CONFIG.validation.max_options_allowed
            )


class TestEnvironmentProfiles:
    """Test environment-specific configuration profiles."""

    def test_development_profile(self):
        """Test development environment profile."""
        with patch.dict(os.environ, {"DMM_ENVIRONMENT": "development"}):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.environment == "development"
            assert config_manager.config.debug_mode is True
            # Development profile should have smaller limits
            assert config_manager.config.validation.max_options_allowed == 10
            assert config_manager.config.session.max_active_sessions == 20

    def test_staging_profile(self):
        """Test staging environment profile."""
        with patch.dict(os.environ, {"DMM_ENVIRONMENT": "staging"}):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.environment == "staging"
            assert config_manager.config.debug_mode is False
            assert config_manager.config.validation.max_options_allowed == 15
            assert config_manager.config.session.max_active_sessions == 50

    def test_production_profile(self):
        """Test production environment profile (default)."""
        # Reset singleton state for test isolation
        ConfigManager._instance = None

        with patch.dict(os.environ, {"DMM_ENVIRONMENT": "production"}):
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            assert config_manager.config.environment == "production"
            assert config_manager.config.debug_mode is False
            # Should use default values
            assert config_manager.config.validation.max_options_allowed == 20
            assert config_manager.config.session.max_active_sessions == 100


class TestConfigurationFileLoading:
    """Test configuration loading from JSON/YAML files."""

    def test_json_config_loading(self):
        """Test loading configuration from JSON file."""
        # Reset singleton state for test isolation
        ConfigManager._instance = None

        config_data = {
            "validation": {"max_options_allowed": 25},
            "session": {"max_active_sessions": 150},
            "backend": {"bedrock_model": "anthropic.claude-3-opus-20240229-v1:0"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Patch the config paths to include our test file
            with patch.object(ConfigManager, "_load_from_files") as mock_load:

                def mock_load_impl(config_data_dict):
                    with open(config_file) as f:
                        file_config = json.load(f)
                    # Note: Can't call self._merge_config in mock, just update config_data_dict
                    config_data_dict.update(file_config)

                mock_load.side_effect = mock_load_impl

                config_manager = ConfigManager()
                config_manager.reload_configuration()

                assert config_manager.config.validation.max_options_allowed == 25
                assert config_manager.config.session.max_active_sessions == 150
                assert (
                    config_manager.config.backend.bedrock_model
                    == "anthropic.claude-3-opus-20240229-v1:0"
                )
        finally:
            os.unlink(config_file)


class TestBackwardCompatibility:
    """Test backward compatibility with existing constants."""

    def test_validation_limits_compatibility(self):
        """Test that ValidationLimits still works as before."""
        # Test accessing properties
        assert ValidationLimits.MAX_OPTIONS_ALLOWED > 0
        assert ValidationLimits.MAX_TOPIC_LENGTH > 0
        assert ValidationLimits.MIN_CRITERION_WEIGHT > 0
        assert ValidationLimits.MAX_CRITERION_WEIGHT > ValidationLimits.MIN_CRITERION_WEIGHT

        # Test that values can change with environment variables
        # Reset singleton state first
        ConfigManager._instance = None

        with patch.dict(
            os.environ, {"DMM_MAX_OPTIONS_ALLOWED": "35", "DMM_ENVIRONMENT": "production"}
        ):
            # Create a new config manager to pick up the env var (production profile won't override)
            new_config_manager = ConfigManager()
            new_config_manager.reload_configuration()

            # Update the global config reference used by constants
            from decision_matrix_mcp.config import config

            config._config = new_config_manager._config

            # The constant should reflect the environment variable override (35)
            assert ValidationLimits.MAX_OPTIONS_ALLOWED == 35

    def test_session_limits_compatibility(self):
        """Test that SessionLimits still works as before."""
        assert SessionLimits.MAX_ACTIVE_SESSIONS > 0
        assert SessionLimits.LRU_EVICTION_BATCH_SIZE > 0
        assert SessionLimits.DEFAULT_MAX_SESSIONS > 0
        assert SessionLimits.DEFAULT_SESSION_TTL_HOURS > 0
        assert SessionLimits.DEFAULT_CLEANUP_INTERVAL_MINUTES > 0

        # Test that values can change with environment variables
        # Reset singleton state first
        ConfigManager._instance = None

        with patch.dict(
            os.environ, {"DMM_MAX_ACTIVE_SESSIONS": "250", "DMM_ENVIRONMENT": "production"}
        ):
            new_config_manager = ConfigManager()
            new_config_manager.reload_configuration()

            # Update the global config reference used by constants
            from decision_matrix_mcp.config import config

            config._config = new_config_manager._config

            assert SessionLimits.MAX_ACTIVE_SESSIONS == 250


class TestRuntimeConfigurationUpdates:
    """Test runtime configuration updates."""

    def test_update_config_method(self):
        """Test updating configuration at runtime."""
        config_manager = ConfigManager()

        # Update some configuration values
        config_manager.update_config(
            validation__max_options_allowed=40,
            performance__request_timeout_seconds=60.0,
            backend__bedrock_model="anthropic.claude-3-haiku-20240307-v1:0",
        )

        assert config_manager.config.validation.max_options_allowed == 40
        assert config_manager.config.performance.request_timeout_seconds == 60.0
        assert (
            config_manager.config.backend.bedrock_model == "anthropic.claude-3-haiku-20240307-v1:0"
        )

    def test_invalid_runtime_update(self):
        """Test handling of invalid runtime configuration updates."""
        config_manager = ConfigManager()

        # Try to set an invalid configuration
        with pytest.raises(ConfigValidationError):
            config_manager.update_config(
                validation__max_options_allowed=-5,  # Invalid: negative value
            )


class TestConfigValidator:
    """Test the ConfigValidator class."""

    def test_system_resource_validation(self):
        """Test system resource validation."""
        # Mock psutil for testing
        with patch("decision_matrix_mcp.config.validation.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value.total = 8 * 1024**3  # 8GB
            mock_psutil.cpu_count.return_value = 8

            config = ConfigSchema()
            config.session.max_active_sessions = 10000  # Very large
            config.performance.max_concurrent_evaluations = 50  # Very large

            validator = ConfigValidator()
            validator.validate_config(config)

            # Should generate warnings about resource usage
            assert len(validator.warnings) > 0

    def test_security_validation(self):
        """Test security constraint validation."""
        config = ConfigSchema()
        config.validation.max_description_length = 50000  # Very large
        config.session.max_active_sessions = 5000  # Very large
        config.performance.request_timeout_seconds = 600  # Very long

        validator = ConfigValidator()
        validator.validate_config(config)

        # Should generate security warnings
        assert len(validator.warnings) > 0
        assert any("DoS" in w or "resource" in w.lower() for w in validator.warnings)

    def test_debug_mode_in_production_warning(self):
        """Test warning for debug mode in production."""
        config = ConfigSchema()
        config.environment = "production"
        config.debug_mode = True

        validator = ConfigValidator()
        validator.validate_config(config)

        # Should generate warning about debug mode in production
        assert len(validator.warnings) > 0
        assert any(
            "debug mode" in w.lower() and "production" in w.lower() for w in validator.warnings
        )


class TestConfigManager:
    """Test the ConfigManager class."""

    def test_singleton_behavior(self):
        """Test that ConfigManager is a singleton."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()

        assert manager1 is manager2

    def test_config_summary(self):
        """Test configuration summary generation."""
        config_manager = ConfigManager()
        summary = config_manager.get_config_summary()

        assert "config_version" in summary
        assert "environment" in summary
        assert "debug_mode" in summary
        assert "validation" in summary
        assert "limits_summary" in summary

    def test_export_config(self):
        """Test configuration export."""
        config_manager = ConfigManager()

        # Test JSON export
        json_export = config_manager.export_config("json")
        assert isinstance(json_export, str)
        assert json.loads(json_export)  # Should be valid JSON

    def test_env_var_help(self):
        """Test environment variable help generation."""
        config_manager = ConfigManager()
        help_text = config_manager.get_env_var_help()

        assert isinstance(help_text, dict)
        assert "DMM_MAX_OPTIONS_ALLOWED" in help_text
        assert "DMM_MAX_ACTIVE_SESSIONS" in help_text

        # Check that help includes type information
        for help_str in help_text.values():
            assert "Type:" in help_str
            assert "Path:" in help_str


class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""

    def test_end_to_end_configuration(self):
        """Test complete configuration flow from env vars to usage."""
        # Reset singleton state for test isolation
        ConfigManager._instance = None

        with patch.dict(
            os.environ,
            {
                "DMM_ENVIRONMENT": "development",
                "DMM_MAX_OPTIONS_ALLOWED": "15",
                "DMM_MAX_ACTIVE_SESSIONS": "75",
                "BEDROCK_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
                "DMM_DEBUG_MODE": "true",
            },
        ):
            # Create new config manager
            config_manager = ConfigManager()
            config_manager.reload_configuration()

            # Update the global config reference used by constants
            from decision_matrix_mcp.config import config as global_config

            global_config._config = config_manager._config

            # Test that all values are correctly loaded
            config = config_manager.config
            assert config.environment == "development"
            # NOTE: Development profile overrides env vars, so max_options_allowed = 10 (profile) not 15 (env var)
            assert config.validation.max_options_allowed == 10
            # NOTE: Development profile also overrides max_active_sessions to 20
            assert config.session.max_active_sessions == 20
            assert config.backend.bedrock_model == "anthropic.claude-3-haiku-20240307-v1:0"
            assert config.debug_mode is True

            # Test that constants reflect development profile (same as config object above)
            assert ValidationLimits.MAX_OPTIONS_ALLOWED == 10
            assert SessionLimits.MAX_ACTIVE_SESSIONS == 20

            # Test validation
            validator = ConfigValidator()
            validator.validate_config(config)  # Should not raise
