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

"""Configuration management system for Decision Matrix MCP.

This module provides a comprehensive configuration management system with:
- Environment variable support with DMM_ prefix
- Type validation and conversion
- Sensible defaults for all configurable values
- Configuration file support (JSON/YAML)
- Backward compatibility with existing hardcoded constants
- Runtime configuration validation
"""

from .defaults import DEFAULT_CONFIG
from .manager import ConfigManager, config
from .schema import (
    BackendLimitsConfig,
    ConfigSchema,
    PerformanceLimitsConfig,
    SessionLimitsConfig,
    ValidationLimitsConfig,
)
from .validation import ConfigValidationError, ConfigValidator

__all__ = [
    "DEFAULT_CONFIG",
    "BackendLimitsConfig",
    "ConfigManager",
    "ConfigSchema",
    "ConfigValidationError",
    "ConfigValidator",
    "PerformanceLimitsConfig",
    "SessionLimitsConfig",
    "ValidationLimitsConfig",
    "config",
]
