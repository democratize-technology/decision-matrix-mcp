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
Validation decorators for request validation with consistent error handling.
Eliminates duplication of validation patterns across MCP tool functions.
"""

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from .constants import ValidationLimits
from .session_manager import SessionValidator

# Error message templates for consistent error responses
ERROR_MESSAGES = {
    "session_id": "Invalid session ID",
    "topic": f"Invalid topic: must be a non-empty string under {ValidationLimits.MAX_TOPIC_LENGTH} characters",
    "option_name": f"Invalid option name: must be 1-{ValidationLimits.MAX_OPTION_NAME_LENGTH} characters",
    "name": f"Invalid criterion name: must be 1-{ValidationLimits.MAX_CRITERION_NAME_LENGTH} characters",
    "description": f"Invalid description: must be 1-{ValidationLimits.MAX_DESCRIPTION_LENGTH} characters",
    "weight": f"Invalid weight: must be between {ValidationLimits.MIN_CRITERION_WEIGHT} and {ValidationLimits.MAX_CRITERION_WEIGHT}",
}


def get_error_message(field: str, value: Any = None) -> str:
    """Get the appropriate error message for a field validation failure."""
    base_message = ERROR_MESSAGES.get(field, f"Invalid {field}")

    # For fields that include the value in the error message
    if field == "option_name" and value:
        return f"Invalid option name: '{value}' (must be 1-{ValidationLimits.MAX_OPTION_NAME_LENGTH} characters)"
    elif field == "criterion_name" and value:
        return f"Invalid criterion name: '{value}' ({base_message})"

    return base_message


def validate_request(**validators: Callable[[Any], bool]) -> Callable[[Any], Any]:
    """
    Decorator for request validation with consistent error handling.

    Args:
        **validators: Mapping of field names to validation functions

    Example:
        @validate_request(
            session_id=SessionValidator.validate_session_id,
            name=SessionValidator.validate_criterion_name
        )
        async def add_criterion(request: AddCriterionRequest) -> dict[str, Any]:
            # Function body with validation already handled
    """

    def decorator(
        func: Callable[..., Awaitable[dict[str, Any]]],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        @wraps(func)
        async def wrapper(request: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            # Validate each field
            for field, validator in validators.items():
                value = getattr(request, field, None)

                # Special handling for list validation (e.g., options)
                if field == "options" and isinstance(value, list):
                    # Check list length constraints
                    if not value or len(value) < ValidationLimits.MIN_OPTIONS_REQUIRED:
                        return {
                            "error": "Need at least 2 options to create a meaningful decision matrix"
                        }
                    if len(value) > ValidationLimits.MAX_OPTIONS_ALLOWED:
                        return {
                            "error": f"Too many options (max {ValidationLimits.MAX_OPTIONS_ALLOWED}). Consider grouping similar options."
                        }
                    # Validate each option
                    for option_name in value:
                        if not SessionValidator.validate_option_name(option_name):
                            return {"error": get_error_message("option_name", option_name)}
                else:
                    # Standard validation
                    if not validator(value):
                        return {"error": get_error_message(field, value)}

            # All validations passed, call the original function
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def validate_criteria_spec(criteria_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Validate a list of criterion specifications.

    Args:
        criteria_list: List of criterion dictionaries with name, description, weight

    Returns:
        Error dict if validation fails, None if all valid
    """
    for criterion_spec in criteria_list:
        name = criterion_spec.get("name", "")
        description = criterion_spec.get("description", "")
        weight = criterion_spec.get("weight", 1.0)

        if not SessionValidator.validate_criterion_name(name):
            return {"error": f"Invalid criterion name: '{name}'"}

        if not SessionValidator.validate_description(description):
            return {"error": f"Invalid criterion description for '{name}'"}

        if not SessionValidator.validate_weight(weight):
            return {"error": get_error_message("weight") + f" for '{name}'"}

    return None
