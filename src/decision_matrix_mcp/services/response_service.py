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

"""Response Service - Standardized response formatting.

Handles creation of consistent response structures for all MCP tool outputs.
"""

from datetime import datetime, timezone
import logging
from typing import Any

from ..formatting import DecisionFormatter
from ..models import DecisionSession

logger = logging.getLogger(__name__)


class ResponseService:
    """Service for creating standardized response structures."""

    def __init__(self, formatter: DecisionFormatter) -> None:
        """Initialize ResponseService with formatter.

        Args:
            formatter: DecisionFormatter instance for output formatting
        """
        self.formatter = formatter

    def create_error_response(
        self,
        message: str,
        context: str = "Validation error",
        error_code: str | None = None,
        error_category: str | None = None,
        recovery_suggestion: str | None = None,
        diagnostic_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create standardized error response with enhanced error information.

        Args:
            message: The error message
            context: Context for the error
            error_code: Optional error code for tracking
            error_category: Optional error category (CLIENT_ERROR, SERVER_ERROR, etc.)
            recovery_suggestion: Optional suggestion for error recovery
            diagnostic_context: Optional diagnostic information for debugging

        Returns:
            Error response dictionary with formatted output and enhanced error info
        """
        error_response: dict[str, Any] = {
            "error": message,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if error_code:
            error_response["error_code"] = error_code
        if error_category:
            error_response["error_category"] = error_category
        if recovery_suggestion:
            error_response["recovery_suggestion"] = recovery_suggestion
        if diagnostic_context:
            error_response["diagnostic_context"] = diagnostic_context

        error_response["formatted_output"] = self.formatter.format_error(message, context)
        return error_response

    def create_session_response(
        self,
        session: DecisionSession,
        request: Any,
        criteria_added: list[str],
    ) -> dict[str, Any]:
        """Create standardized session creation response.

        Args:
            session: The created decision session
            request: Original request with topic and options
            criteria_added: List of criteria names that were added

        Returns:
            Response dictionary
        """
        message = f"Decision analysis initialized with {len(request.options)} options"
        if criteria_added:
            message += f" and {len(criteria_added)} criteria"

        response_data = {
            "session_id": session.session_id,
            "topic": request.topic,
            "options": request.options,
            "criteria_added": criteria_added,
            "model_backend": request.model_backend.value,
            "model_name": request.model_name,
            "message": message,
            "next_steps": [
                "add_criterion - Add evaluation criteria",
                "evaluate_options - Run the analysis",
                "get_decision_matrix - See results",
            ],
        }

        # Add formatted output
        response_data["formatted_output"] = self.formatter.format_session_created(response_data)

        return response_data

    def create_criterion_response(self, request: Any, session: DecisionSession) -> dict[str, Any]:
        """Create standardized criterion addition response.

        Args:
            request: Request containing criterion data
            session: Decision session with updated criteria

        Returns:
            Response dictionary
        """
        response_data = {
            "session_id": request.session_id,
            "criterion_added": request.name,
            "description": request.description,
            "weight": request.weight,
            "total_criteria": len(session.criteria),
            "all_criteria": list(session.criteria.keys()),
            "message": f"Added criterion '{request.name}' with weight {request.weight}x",
        }

        # Add formatted output
        response_data["formatted_output"] = self.formatter.format_criterion_added(response_data)

        return response_data

    def create_evaluation_response(
        self,
        session_id: str,
        session: DecisionSession,
        total_scores: int,
        abstentions: int,
        errors: list[str],
    ) -> dict[str, Any]:
        """Create standardized evaluation response.

        Args:
            session_id: ID of the session
            session: Decision session
            total_scores: Number of successful scores
            abstentions: Number of abstentions
            errors: List of error messages

        Returns:
            Response dictionary
        """
        response_data = {
            "session_id": session_id,
            "evaluation_complete": True,
            "summary": {
                "options_evaluated": len(session.options),
                "criteria_used": len(session.criteria),
                "total_evaluations": len(session.options) * len(session.criteria),
                "successful_scores": total_scores,
                "abstentions": abstentions,
                "errors": len(errors),
            },
            "errors": errors if errors else None,
            "message": f"Evaluated {len(session.options)} options across {len(session.criteria)} criteria",
            "next_steps": ["get_decision_matrix - See the complete results"],
        }

        # Add formatted output
        response_data["formatted_output"] = self.formatter.format_evaluation_complete(response_data)

        return response_data

    def create_matrix_response(
        self,
        matrix_result: dict[str, Any],
        session: DecisionSession,
    ) -> dict[str, Any]:
        """Create standardized decision matrix response.

        Args:
            matrix_result: Raw matrix results from session
            session: Decision session for metadata

        Returns:
            Response dictionary with formatted output
        """
        if "error" in matrix_result:
            matrix_result["formatted_output"] = self.formatter.format_error(matrix_result["error"])
            return matrix_result

        # Add session metadata
        matrix_result.update(
            {
                "session_info": {
                    "created_at": session.created_at.isoformat(),
                    "evaluations_run": len(session.evaluations),
                    "total_options": len(session.options),
                    "total_criteria": len(session.criteria),
                },
            },
        )

        # Add formatted output
        matrix_result["formatted_output"] = self.formatter.format_decision_matrix(matrix_result)

        return matrix_result

    def create_option_added_response(
        self,
        request: Any,
        session: DecisionSession,
    ) -> dict[str, Any]:
        """Create standardized option addition response.

        Args:
            request: Request containing option data
            session: Decision session with updated options

        Returns:
            Response dictionary
        """
        response_data = {
            "session_id": request.session_id,
            "option_added": request.option_name,
            "description": request.description,
            "total_options": len(session.options),
            "all_options": list(session.options.keys()),
            "message": f"Added option '{request.option_name}'",
            "next_steps": ["evaluate_options - Re-run evaluation to include new option"],
        }

        # Add formatted output
        response_data["formatted_output"] = self.formatter.format_option_added(response_data)

        return response_data

    def create_sessions_list_response(
        self,
        active_sessions: dict[str, DecisionSession],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Create standardized sessions list response.

        Args:
            active_sessions: Dictionary of active sessions
            stats: Session manager statistics

        Returns:
            Response dictionary
        """
        session_list = []
        for sid, session in active_sessions.items():
            session_list.append(
                {
                    "session_id": sid,
                    "topic": session.topic,
                    "created_at": session.created_at.isoformat(),
                    "options": list(session.options.keys()),
                    "criteria": list(session.criteria.keys()),
                    "evaluations_run": len(session.evaluations),
                    "status": "evaluated" if session.evaluations else "setup",
                },
            )

        response_data = {
            "sessions": session_list,
            "total_active": len(active_sessions),
            "stats": stats,
        }

        # Add formatted output
        response_data["formatted_output"] = self.formatter.format_sessions_list(response_data)

        return response_data

    def create_current_session_response(self, session: DecisionSession | None) -> dict[str, Any]:
        """Create standardized current session response.

        Args:
            session: Current session or None if no active sessions

        Returns:
            Response dictionary
        """
        if not session:
            return {
                "session": None,
                "message": "No active sessions found",
                "formatted_output": self.formatter.format_error(
                    "No active sessions",
                    "No current session",
                ),
            }

        response_data = {
            "session_id": session.session_id,
            "topic": session.topic,
            "created_at": session.created_at.isoformat(),
            "options": list(session.options),  # options is a dict, keys are the option names
            "criteria": list(session.criteria),  # criteria is a dict, keys are the criterion names
            "evaluations_run": len(session.evaluations),
            "status": "evaluated" if len(session.evaluations) > 0 else "pending",
            "model_backend": (
                session.model_backend.value if hasattr(session, "model_backend") else "bedrock"
            ),
            "message": f"Current session: {session.topic}",
        }

        # Add formatted output
        response_data["formatted_output"] = self.formatter.format_session_summary(session)

        return response_data

    def create_bedrock_test_response(self, test_result: dict[str, Any]) -> dict[str, Any]:
        """Create standardized Bedrock connection test response.

        Args:
            test_result: Raw test results from orchestrator

        Returns:
            Response dictionary with formatted output
        """
        # Format for LLM consumption
        if test_result["status"] == "ok":
            formatted_output = f"""# ✅ Bedrock Connection Test: SUCCESS

**Status**: {test_result["status"].upper()}
**Region**: {test_result["region"]}
**Model Tested**: {test_result["model_tested"]}
**Response Length**: {test_result["response_length"]} characters

{test_result["message"]}"""
        else:
            formatted_output = self.formatter.format_error(
                f"❌ Bedrock connection failed: {test_result['error']}",
                f"Suggestion: {test_result.get('suggestion', 'Check AWS configuration')}",
            )

        test_result["formatted_output"] = formatted_output
        return test_result
