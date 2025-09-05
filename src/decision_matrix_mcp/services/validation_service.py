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

"""Validation Service - Centralized validation logic.

Handles session validation, input validation, and prerequisite checking.
"""

import logging
from typing import Any

from ..models import DecisionSession
from ..session_manager import SessionValidator

logger = logging.getLogger(__name__)


class ValidationService:
    """Centralized validation service for all decision analysis operations."""

    def __init__(self) -> None:
        """Initialize ValidationService."""
        self.session_validator = SessionValidator

    def validate_session_id(self, session_id: str) -> bool:
        """Validate session ID format.

        Args:
            session_id: Session ID to validate

        Returns:
            True if valid format, False otherwise
        """
        return self.session_validator.validate_session_id(session_id)

    def validate_session_exists(
        self,
        session_id: str,
        session_manager: Any,
    ) -> tuple[DecisionSession | None, dict[str, Any] | None]:
        """Validate session exists and return it.

        Args:
            session_id: ID of the session to retrieve
            session_manager: Session manager instance

        Returns:
            Tuple of (session, None) if successful, or (None, error_dict) if failed
        """
        # Validate session ID format first
        if not self.validate_session_id(session_id):
            error_dict = {
                "error": "Invalid session ID",
            }
            return None, error_dict

        session = session_manager.get_session(session_id)
        if not session:
            error_dict = {
                "error": "Session not found or expired",
            }
            return None, error_dict

        return session, None

    def validate_evaluation_prerequisites(self, session: DecisionSession) -> dict[str, Any] | None:
        """Validate that a session is ready for evaluation.

        Args:
            session: The decision session to validate

        Returns:
            Error dict if validation fails, None if valid
        """
        if not session.options:
            return {
                "error": "No options to evaluate. Add options first.",
                "validation_context": "prerequisites",
            }

        if not session.criteria:
            return {
                "error": "No criteria defined. Add criteria first.",
                "validation_context": "prerequisites",
            }

        return None

    def validate_criterion_name(self, name: str) -> bool:
        """Validate criterion name.

        Args:
            name: Criterion name to validate

        Returns:
            True if valid, False otherwise
        """
        return self.session_validator.validate_criterion_name(name)

    def validate_option_name(self, name: str) -> bool:
        """Validate option name.

        Args:
            name: Option name to validate

        Returns:
            True if valid, False otherwise
        """
        return self.session_validator.validate_option_name(name)

    def validate_topic(self, topic: str) -> bool:
        """Validate topic string.

        Args:
            topic: Topic to validate

        Returns:
            True if valid, False otherwise
        """
        return self.session_validator.validate_topic(topic)

    def validate_description(self, description: str) -> bool:
        """Validate description string.

        Args:
            description: Description to validate

        Returns:
            True if valid, False otherwise
        """
        return self.session_validator.validate_description(description)

    def validate_weight(self, weight: float) -> bool:
        """Validate criterion weight.

        Args:
            weight: Weight value to validate

        Returns:
            True if valid, False otherwise
        """
        return self.session_validator.validate_weight(weight)

    def validate_criteria_spec(self, criteria_spec: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Validate initial criteria specification.

        Args:
            criteria_spec: List of criterion specifications

        Returns:
            Error dict if validation fails, None if valid
        """
        # Import here to avoid circular imports
        from ..validation_decorators import validate_criteria_spec

        return validate_criteria_spec(criteria_spec)

    def validate_criterion_exists(self, session: DecisionSession, criterion_name: str) -> bool:
        """Check if criterion already exists in session.

        Args:
            session: Decision session to check
            criterion_name: Name of criterion to check

        Returns:
            True if criterion exists, False otherwise
        """
        return criterion_name in session.criteria

    def validate_option_exists(self, session: DecisionSession, option_name: str) -> bool:
        """Check if option already exists in session.

        Args:
            session: Decision session to check
            option_name: Name of option to check

        Returns:
            True if option exists, False otherwise
        """
        return option_name in session.options
