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

"""Helper functions to reduce complexity in main module."""

import logging
from typing import Any

from .models import DecisionSession, Score

logger = logging.getLogger(__name__)


def validate_evaluation_prerequisites(session: DecisionSession) -> dict[str, Any] | None:
    """Validate that a session is ready for evaluation.

    Args:
        session: The decision session to validate

    Returns:
        Error response dict if validation fails, None if valid
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


def process_evaluation_results(
    evaluation_results: dict[str, dict[str, tuple[float | None, str]]],
    session: DecisionSession,
) -> tuple[int, int, list[str]]:
    """Process evaluation results and update session scores.

    Args:
        evaluation_results: Results from orchestrator evaluation
        session: Decision session to update

    Returns:
        Tuple of (total_scores, abstentions, errors)
    """
    total_scores = 0
    abstentions = 0
    errors = []

    for criterion_name, option_results in evaluation_results.items():
        for option_name, (score, justification) in option_results.items():
            score_obj = Score(
                criterion_name=criterion_name,
                option_name=option_name,
                score=score,
                justification=justification,
            )

            if option_name in session.options:
                session.options[option_name].add_score(score_obj)

                if "Error:" in justification:
                    errors.append(f"{criterion_name}→{option_name}: {justification}")
                elif score is None:
                    abstentions += 1
                else:
                    total_scores += 1

    return total_scores, abstentions, errors


def create_evaluation_response(
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
    return {
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


def create_error_response(
    error_msg: str,
    context: str = "Validation error",
    formatter: Any | None = None,
) -> dict[str, Any]:
    """Create standardized error response.

    Args:
        error_msg: The error message
        context: Context for the error
        formatter: Optional formatter for enhanced output

    Returns:
        Error response dictionary
    """
    error_response = {"error": error_msg}

    if formatter:
        error_response["formatted_output"] = formatter.format_error(error_msg, context)

    return error_response


def process_initial_criteria(request: Any, session: DecisionSession) -> list[str]:
    """Process and add initial criteria to a session.

    Args:
        request: Request containing initial_criteria and model settings
        session: Decision session to add criteria to

    Returns:
        List of criteria names that were added
    """
    criteria_added = []

    if not request.initial_criteria:
        return criteria_added

    for criterion_spec in request.initial_criteria:
        name = criterion_spec.get("name", "")
        description = criterion_spec.get("description", "")
        weight = criterion_spec.get("weight", 1.0)

        from .models import Criterion

        criterion = Criterion(
            name=name,
            description=description,
            weight=weight,
            model_backend=request.model_backend,
            model_name=request.model_name,
            temperature=criterion_spec.get("temperature", request.temperature),
        )

        session.add_criterion(criterion)
        criteria_added.append(name)

    return criteria_added


def create_session_response(
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

    return {
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


def create_criterion_from_request(request: Any, session: DecisionSession) -> Any:
    """Create a Criterion object from request data.

    Args:
        request: Request containing criterion parameters
        session: Decision session for default temperature

    Returns:
        Criterion object
    """
    from .models import Criterion

    criterion = Criterion(
        name=request.name,
        description=request.description,
        weight=request.weight,
        model_backend=request.model_backend,
        model_name=request.model_name,
        temperature=(
            request.temperature if request.temperature is not None else session.default_temperature
        ),
    )

    # Override system prompt if provided
    if request.custom_prompt:
        criterion.system_prompt = request.custom_prompt

    return criterion


def create_criterion_response(request: Any, session: DecisionSession) -> dict[str, Any]:
    """Create standardized criterion addition response.

    Args:
        request: Request containing criterion data
        session: Decision session with updated criteria

    Returns:
        Response dictionary
    """
    return {
        "session_id": request.session_id,
        "criterion_added": request.name,
        "description": request.description,
        "weight": request.weight,
        "total_criteria": len(session.criteria),
        "all_criteria": list(session.criteria.keys()),
        "message": f"Added criterion '{request.name}' with weight {request.weight}x",
    }
