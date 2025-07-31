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

from .constants import ValidationLimits
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
from .session_manager import SessionManager, SessionValidator
from .validation_decorators import validate_criteria_spec, validate_request

__all__ = [
    "main",
    "mcp",
    "create_server_components",
    "ServerComponents",
    "get_session_or_error",
    "get_server_components",
    # Request models
    "StartDecisionAnalysisRequest",
    "AddCriterionRequest",
    "EvaluateOptionsRequest",
    "GetDecisionMatrixRequest",
    "AddOptionRequest",
    # Tool functions
    "start_decision_analysis",
    "add_criterion",
    "evaluate_options",
    "get_decision_matrix",
    "add_option",
    "list_sessions",
    "clear_all_sessions",
]

# Configure logging to stderr only - NEVER stdout in MCP servers
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class ServerComponents:
    """Container for server dependencies with proper lifecycle management."""

    def __init__(self, orchestrator: DecisionOrchestrator | None = None,
                 session_manager: SessionManager | None = None):
        """Initialize server components.

        Args:
            orchestrator: Decision orchestrator instance
            session_manager: Session manager instance
        """
        self.orchestrator = orchestrator or DecisionOrchestrator()
        self.session_manager = session_manager or SessionManager()

    def cleanup(self) -> None:
        """Clean up server resources."""
        # Clear all sessions to prevent memory leaks
        self.session_manager.clear_all_sessions()


def create_server_components() -> ServerComponents:
    """Create server components for dependency injection.

    Returns:
        ServerComponents: Container with orchestrator and session_manager
    """
    return ServerComponents()


# Create FastMCP instance with component factory
mcp = FastMCP("decision-matrix")

# Server components - injected per request context
_server_components: ServerComponents | None = None


def get_server_components() -> ServerComponents:
    """Get or create server components.

    Returns:
        ServerComponents: Server component container
    """
    global _server_components
    if _server_components is None:
        _server_components = create_server_components()
    return _server_components


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


def get_session_or_error(session_id: str, components: ServerComponents) -> tuple[DecisionSession | None, dict[str, Any] | None]:
    """
    Get session or return error dict for consistent session validation.

    Args:
        session_id: ID of the session to retrieve
        components: Server components container

    Returns:
        Tuple of (session, None) if successful, or (None, error_dict) if failed
    """
    # Validate session ID format first
    if not SessionValidator.validate_session_id(session_id):
        return None, {"error": "Invalid session ID format"}

    # Get session from manager
    session = components.session_manager.get_session(session_id)
    if not session:
        return None, {"error": f"Session {session_id} not found or expired"}

    return session, None


@mcp.tool(
    description="When facing multiple options and need structured evaluation - create a decision matrix to systematically compare choices across weighted criteria"
)
@validate_request(
    topic=SessionValidator.validate_topic,
    options=SessionValidator.validate_option_name,  # Special handling in decorator for lists
)
async def start_decision_analysis(request: StartDecisionAnalysisRequest) -> dict[str, Any]:
    """Initialize a new decision analysis session with options and optional criteria"""

    components = get_server_components()

    try:
        session = components.session_manager.create_session(
            topic=request.topic, initial_options=request.options
        )

        criteria_added = []
        if request.initial_criteria:
            # Validate all criteria specs
            validation_error = validate_criteria_spec(request.initial_criteria)
            if validation_error:
                return validation_error

            # Add validated criteria
            for criterion_spec in request.initial_criteria:
                name = criterion_spec.get("name", "")
                description = criterion_spec.get("description", "")
                weight = criterion_spec.get("weight", 1.0)

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
@validate_request(
    session_id=SessionValidator.validate_session_id,
    name=SessionValidator.validate_criterion_name,
    description=SessionValidator.validate_description,
    weight=SessionValidator.validate_weight,
)
async def add_criterion(request: AddCriterionRequest) -> dict[str, Any]:
    """Add a new evaluation criterion to an existing decision session"""

    components = get_server_components()

    # Get session
    session, error = get_session_or_error(request.session_id, components)
    if error:
        return error

    # Session validation guard
    assert session is not None, "Session should not be None after successful get_session_or_error"

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
@validate_request(session_id=SessionValidator.validate_session_id)
async def evaluate_options(request: EvaluateOptionsRequest) -> dict[str, Any]:
    """Evaluate all options across all criteria using parallel thread orchestration"""

    components = get_server_components()

    # Get session
    session, error = get_session_or_error(request.session_id, components)
    if error:
        return error

    # Session validation guard
    assert session is not None, "Session should not be None after successful get_session_or_error"

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
        evaluation_results = await components.orchestrator.evaluate_options_across_criteria(
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

                    if "Error:" in justification:
                        errors.append(f"{criterion_name}→{option_name}: {justification}")
                    elif score is None:
                        abstentions += 1
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
@validate_request(session_id=SessionValidator.validate_session_id)
async def get_decision_matrix(request: GetDecisionMatrixRequest) -> dict[str, Any]:
    """Get the complete decision matrix with scores, rankings, and recommendations"""

    components = get_server_components()

    # Get session
    session, error = get_session_or_error(request.session_id, components)
    if error:
        return error

    # Session validation guard
    assert session is not None, "Session should not be None after successful get_session_or_error"

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
@validate_request(
    session_id=SessionValidator.validate_session_id,
    option_name=SessionValidator.validate_option_name,
)
async def add_option(request: AddOptionRequest) -> dict[str, Any]:
    """Add a new option to an existing decision analysis"""

    components = get_server_components()

    # Get session
    session, error = get_session_or_error(request.session_id, components)
    if error:
        return error

    # Session validation guard
    assert session is not None, "Session should not be None after successful get_session_or_error"

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
    components = get_server_components()

    try:
        active_sessions = components.session_manager.list_active_sessions()

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
        stats = components.session_manager.get_stats()

        return {"sessions": session_list, "total_active": len(active_sessions), "stats": stats}

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return {"error": f"Failed to list sessions: {str(e)}"}


@mcp.tool(description="Clear all active decision analysis sessions")
async def clear_all_sessions() -> dict[str, Any]:
    """Clear all active sessions from the session manager"""
    components = get_server_components()

    try:
        active_sessions = components.session_manager.list_active_sessions()
        cleared_count = 0

        for session_id in list(active_sessions.keys()):
            if components.session_manager.remove_session(session_id):
                cleared_count += 1

        return {
            "cleared": cleared_count,
            "message": f"Cleared {cleared_count} active sessions",
            "stats": components.session_manager.get_stats(),
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
