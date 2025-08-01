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
