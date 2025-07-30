#!/usr/bin/env python3
"""
Decision Matrix MCP Server
Structured decision analysis using thread orchestration for parallel criterion evaluation

CRITICAL: This server uses stdio transport for MCP protocol communication.
- stdout is reserved for MCP JSON-RPC messages
- All logging/debug output must go to stderr or files
- Never print() to stdout in MCP server code
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .exceptions import (
    ConfigurationError,
    DecisionMatrixError,
    LLMBackendError,
    ResourceLimitError,
    SessionError,
    ValidationError,
)
from .models import Criterion, DecisionSession, ModelBackend, Option, Score
from .orchestrator import DecisionOrchestrator
from .session_manager import SessionValidator, session_manager

__all__ = ["main", "mcp"]

# Configure logging to stderr only - NEVER stdout in MCP servers
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("decision-matrix")

orchestrator = DecisionOrchestrator()


class StartDecisionAnalysisRequest(BaseModel):
    """Request to start a new decision analysis"""

    topic: str = Field(description="The decision topic or question to analyze")
    options: list[str] = Field(description="List of options to evaluate")
    initial_criteria: list[dict[str, Any]] | None = Field(
        default=None, description="Optional initial criteria with name, description, and weight"
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK, description="LLM backend to use for evaluations"
    )
    model_name: str | None = Field(default=None, description="Specific model to use")


@mcp.tool(
    description="When facing multiple options and need structured evaluation - create a decision matrix to systematically compare choices across weighted criteria"
)
async def start_decision_analysis(request: StartDecisionAnalysisRequest) -> dict[str, Any]:
    """Initialize a new decision analysis session with options and optional criteria"""

    if not SessionValidator.validate_topic(request.topic):
        return {"error": "Invalid topic: must be a non-empty string under 500 characters"}

    if not request.options or len(request.options) < 2:
        return {"error": "Need at least 2 options to create a meaningful decision matrix"}

    if len(request.options) > 20:
        return {"error": "Too many options (max 20). Consider grouping similar options."}

    for option_name in request.options:
        if not SessionValidator.validate_option_name(option_name):
            return {"error": f"Invalid option name: '{option_name}' (must be 1-200 characters)"}

    try:
        session = session_manager.create_session(
            topic=request.topic, initial_options=request.options
        )

        criteria_added = []
        if request.initial_criteria:
            for criterion_spec in request.initial_criteria:
                name = criterion_spec.get("name", "")
                description = criterion_spec.get("description", "")
                weight = criterion_spec.get("weight", 1.0)

                if not SessionValidator.validate_criterion_name(name):
                    return {"error": f"Invalid criterion name: '{name}'"}

                if not SessionValidator.validate_description(description):
                    return {"error": f"Invalid criterion description for '{name}'"}

                if not SessionValidator.validate_weight(weight):
                    return {"error": f"Invalid weight for '{name}': must be 0.1-10.0"}

                criterion = Criterion(
                    name=name,
                    description=description,
                    weight=weight,
                    model_backend=request.model_backend,
                    model_name=request.model_name,
                )

                session.add_criterion(criterion)
                criteria_added.append(name)

        return {
            "session_id": session.session_id,
            "topic": request.topic,
            "options": request.options,
            "criteria_added": criteria_added,
            "model_backend": request.model_backend.value,
            "model_name": request.model_name,
            "message": f"Decision analysis initialized with {len(request.options)} options"
            + (f" and {len(criteria_added)} criteria" if criteria_added else ""),
            "next_steps": [
                "add_criterion - Add evaluation criteria",
                "evaluate_options - Run the analysis",
                "get_decision_matrix - See results",
            ],
        }

    except ValidationError as e:
        logger.warning(f"Invalid input for decision session: {e}")
        return {"error": e.user_message}
    except ResourceLimitError as e:
        logger.warning(f"Resource limit exceeded: {e}")
        return {"error": e.user_message}
    except Exception:
        logger.exception("Unexpected error creating decision session")
        return {"error": "Failed to create session due to an unexpected error"}


class AddCriterionRequest(BaseModel):
    """Request to add an evaluation criterion"""

    session_id: str = Field(description="Session ID to add criterion to")
    name: str = Field(description="Name of the criterion (e.g., 'performance', 'cost')")
    description: str = Field(description="What this criterion evaluates")
    weight: float = Field(default=1.0, description="Importance weight (0.1-10.0, default 1.0)")
    custom_prompt: str | None = Field(
        default=None, description="Custom evaluation prompt for this criterion"
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK, description="LLM backend to use for this criterion"
    )
    model_name: str | None = Field(default=None, description="Specific model to use")


@mcp.tool(
    description="When you identify another factor to consider - add evaluation criteria with weights to structure your decision analysis"
)
async def add_criterion(request: AddCriterionRequest) -> dict[str, Any]:
    """Add a new evaluation criterion to an existing decision session"""

    # Validate inputs
    if not SessionValidator.validate_session_id(request.session_id):
        return {"error": "Invalid session ID"}

    if not SessionValidator.validate_criterion_name(request.name):
        return {"error": "Invalid criterion name: must be 1-100 characters"}

    if not SessionValidator.validate_description(request.description):
        return {"error": "Invalid description: must be 1-1000 characters"}

    if not SessionValidator.validate_weight(request.weight):
        return {"error": "Invalid weight: must be between 0.1 and 10.0"}

    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found or expired"}

    # Check if criterion already exists
    if request.name in session.criteria:
        return {"error": f"Criterion '{request.name}' already exists"}

    try:
        # Create criterion
        criterion = Criterion(
            name=request.name,
            description=request.description,
            weight=request.weight,
            model_backend=request.model_backend,
            model_name=request.model_name,
        )

        # Override system prompt if provided
        if request.custom_prompt:
            criterion.system_prompt = request.custom_prompt

        # Add to session
        session.add_criterion(criterion)

        return {
            "session_id": request.session_id,
            "criterion_added": request.name,
            "description": request.description,
            "weight": request.weight,
            "total_criteria": len(session.criteria),
            "all_criteria": list(session.criteria.keys()),
            "message": f"Added criterion '{request.name}' with weight {request.weight}x",
        }

    except SessionError as e:
        logger.warning(f"Session error when adding criterion: {e}")
        return {"error": e.user_message}
    except ValidationError as e:
        logger.warning(f"Invalid criterion input: {e}")
        return {"error": e.user_message}
    except Exception:
        logger.exception("Unexpected error adding criterion")
        return {"error": "Failed to add criterion due to an unexpected error"}


class EvaluateOptionsRequest(BaseModel):
    """Request to evaluate all options against all criteria"""

    session_id: str = Field(description="Session ID for evaluation")


@mcp.tool(
    description="When ready to score your options systematically - run parallel evaluation where each criterion scores every option independently"
)
async def evaluate_options(request: EvaluateOptionsRequest) -> dict[str, Any]:
    """Evaluate all options across all criteria using parallel thread orchestration"""

    # Validate session ID
    if not SessionValidator.validate_session_id(request.session_id):
        return {"error": "Invalid session ID"}

    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found or expired"}

    # Check prerequisites
    if not session.options:
        return {"error": "No options to evaluate. Add options first."}

    if not session.criteria:
        return {"error": "No criteria defined. Add criteria first."}

    try:
        logger.info(
            f"Starting evaluation: {len(session.options)} options × {len(session.criteria)} criteria"
        )

        # Run parallel evaluation using orchestrator
        evaluation_results = await orchestrator.evaluate_options_across_criteria(
            session.threads, list(session.options.values())
        )

        # Process results and update session
        total_scores = 0
        abstentions = 0
        errors = []

        for criterion_name, option_results in evaluation_results.items():
            for option_name, (score, justification) in option_results.items():
                # Create Score object
                score_obj = Score(
                    criterion_name=criterion_name,
                    option_name=option_name,
                    score=score,
                    justification=justification,
                )

                # Add to option
                if option_name in session.options:
                    session.options[option_name].add_score(score_obj)

                    if score is None:
                        abstentions += 1
                    elif "Error:" in justification:
                        errors.append(f"{criterion_name}→{option_name}: {justification}")
                    else:
                        total_scores += 1

        # Record evaluation in session history
        session.record_evaluation(
            {
                "evaluation_results": evaluation_results,
                "total_scores": total_scores,
                "abstentions": abstentions,
                "errors": len(errors),
            }
        )

        return {
            "session_id": request.session_id,
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

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}


class GetDecisionMatrixRequest(BaseModel):
    """Request to get the complete decision matrix"""

    session_id: str = Field(description="Session ID to get matrix for")


@mcp.tool(
    description="When you need the complete picture - see the scored matrix with weighted totals, rankings, and clear recommendations"
)
async def get_decision_matrix(request: GetDecisionMatrixRequest) -> dict[str, Any]:
    """Get the complete decision matrix with scores, rankings, and recommendations"""

    # Validate session ID
    if not SessionValidator.validate_session_id(request.session_id):
        return {"error": "Invalid session ID"}

    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found or expired"}

    try:
        # Generate decision matrix
        matrix_result = session.get_decision_matrix()

        if "error" in matrix_result:
            return matrix_result

        # Add session metadata
        matrix_result.update(
            {
                "session_info": {
                    "created_at": session.created_at.isoformat(),
                    "evaluations_run": len(session.evaluations),
                    "total_options": len(session.options),
                    "total_criteria": len(session.criteria),
                }
            }
        )

        return matrix_result

    except Exception as e:
        logger.error(f"Error generating decision matrix: {e}")
        return {"error": f"Failed to generate matrix: {str(e)}"}


class AddOptionRequest(BaseModel):
    """Request to add a new option to existing analysis"""

    session_id: str = Field(description="Session ID to add option to")
    option_name: str = Field(description="Name of the new option")
    description: str | None = Field(default=None, description="Optional description")


@mcp.tool(
    description="When new alternatives emerge during analysis - add additional options to your existing decision matrix"
)
async def add_option(request: AddOptionRequest) -> dict[str, Any]:
    """Add a new option to an existing decision analysis"""

    # Validate inputs
    if not SessionValidator.validate_session_id(request.session_id):
        return {"error": "Invalid session ID"}

    if not SessionValidator.validate_option_name(request.option_name):
        return {"error": "Invalid option name: must be 1-200 characters"}

    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found or expired"}

    # Check if option already exists
    if request.option_name in session.options:
        return {"error": f"Option '{request.option_name}' already exists"}

    try:
        # Add option
        session.add_option(request.option_name, request.description)

        return {
            "session_id": request.session_id,
            "option_added": request.option_name,
            "description": request.description,
            "total_options": len(session.options),
            "all_options": list(session.options.keys()),
            "message": f"Added option '{request.option_name}'",
            "next_steps": ["evaluate_options - Re-run evaluation to include new option"],
        }

    except Exception as e:
        logger.error(f"Error adding option: {e}")
        return {"error": f"Failed to add option: {str(e)}"}


@mcp.tool(description="List all active decision analysis sessions")
async def list_sessions() -> dict[str, Any]:
    """List all active decision analysis sessions"""
    try:
        active_sessions = session_manager.list_active_sessions()

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
                }
            )

        # Get session manager stats
        stats = session_manager.get_stats()

        return {"sessions": session_list, "total_active": len(active_sessions), "stats": stats}

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return {"error": f"Failed to list sessions: {str(e)}"}


@mcp.tool(description="Clear all active decision analysis sessions")
async def clear_all_sessions() -> dict[str, Any]:
    """Clear all active sessions from the session manager"""
    try:
        active_sessions = session_manager.list_active_sessions()
        cleared_count = 0

        for session_id in list(active_sessions.keys()):
            if session_manager.remove_session(session_id):
                cleared_count += 1

        return {
            "cleared": cleared_count,
            "message": f"Cleared {cleared_count} active sessions",
            "stats": session_manager.get_stats(),
        }

    except Exception as e:
        logger.error(f"Error clearing sessions: {e}")
        return {"error": f"Failed to clear sessions: {str(e)}"}


def main() -> None:
    """Run the Decision Matrix MCP server"""
    try:
        logger.info("Starting Decision Matrix MCP server...")
        logger.debug("Initializing FastMCP...")

        # Test server creation
        logger.debug(f"MCP server created: {mcp}")
        logger.info("Starting MCP server run...")

        mcp.run()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except (BrokenPipeError, ConnectionResetError):
        # Normal disconnection from client
        logger.debug("Client disconnected normally")
    except Exception as e:
        # Check if it's a stdio disconnection wrapped in ExceptionGroup
        error_str = str(e).lower()
        if "brokenresourceerror" in error_str or "broken pipe" in error_str:
            logger.debug("Client disconnected (stdio closed)")
        else:
            logger.error(f"Server error: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise


if __name__ == "__main__":
    logger.debug("Module started as main")
    main()
