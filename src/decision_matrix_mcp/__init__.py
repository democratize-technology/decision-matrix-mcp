#!/usr/bin/env python3
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

"""Decision Matrix MCP Server.

Structured decision analysis using thread orchestration for parallel criterion evaluation.

CRITICAL: This server uses stdio transport for MCP protocol communication.
- stdout is reserved for MCP JSON-RPC messages
- All logging/debug output must go to stderr or files
- Never logger.info() to stdout in MCP server code
"""

import asyncio
from datetime import datetime, timezone
import logging
import sys
import threading
import traceback
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field  # type: ignore[import-not-found]

from .constants import ValidationLimits
from .dependency_injection import ServiceContainer
from .exceptions import (
    ConfigurationError,
    DecisionMatrixError,
    LLMBackendError,
    ResourceLimitError,
    SessionError,
    ValidationError,
)
from .formatting import DecisionFormatter
from .models import Criterion, DecisionSession, ModelBackend, Option, Score
from .orchestrator import DecisionOrchestrator
from .session_manager import SessionManager, SessionValidator
from .validation_decorators import (
    ERROR_MESSAGES,
    ValidationErrorFormatter,
    validate_criteria_spec,
    validate_request,
)

__all__ = [
    "AddCriterionRequest",
    "AddOptionRequest",
    "EvaluateOptionsRequest",
    "GetDecisionMatrixRequest",
    "ServerComponents",
    "StartDecisionAnalysisRequest",
    "add_criterion",
    "add_option",
    "clear_all_sessions",
    "create_server_components",
    "current_session",
    "evaluate_options",
    "get_decision_matrix",
    "get_server_components",
    "get_session_or_error",
    "list_sessions",
    "main",
    "mcp",
    "start_decision_analysis",
    "test_aws_bedrock_connection",
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class ServerComponents:
    """Legacy wrapper for service container - maintained for compatibility."""

    def __init__(self, service_container: ServiceContainer | None = None) -> None:
        self._container = service_container or ServiceContainer()
        self._container.initialize()

        # Expose legacy interface
        self.orchestrator = self._container.get_orchestrator()
        self.session_manager = self._container.get_session_manager()
        self.formatter = self._container.get_formatter()

        # Service layer access
        self.decision_service = self._container.get_decision_service()
        self.validation_service = self._container.get_validation_service()
        self.response_service = self._container.get_response_service()
        self.error_handler = self._container.get_error_handler()

    def cleanup(self) -> None:
        """Clean up all server resources."""
        self._container.cleanup()


def create_server_components() -> ServerComponents:
    """Create and return a new ServerComponents instance.

    Returns:
        ServerComponents: Configured server components container
    """
    return ServerComponents()


# Create FastMCP instance with component factory
mcp = FastMCP("decision-matrix")

# Server components - created once at startup with thread-safe initialization
_server_components: ServerComponents | None = None
_server_components_lock = threading.Lock()


def get_server_components() -> ServerComponents:
    """Get server components with thread-safe lazy initialization."""
    global _server_components  # noqa: PLW0603 - Intentional singleton pattern for thread safety

    # Fast path: if already initialized, return immediately
    if _server_components is not None:
        return _server_components

    # Slow path: initialize with double-checked locking pattern
    with _server_components_lock:
        # Check again inside the lock in case another thread initialized it
        if _server_components is None:
            _server_components = create_server_components()
            logger.info("Server components initialized")
        return _server_components


def initialize_server_components() -> None:
    """Explicitly initialize server components (optional, for eager initialization)."""
    get_server_components()  # This will initialize if needed


class StartDecisionAnalysisRequest(BaseModel):
    """Request to start a new decision analysis."""

    topic: str = Field(description="The decision topic or question to analyze")
    options: list[str] = Field(description="List of options to evaluate")
    initial_criteria: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional initial criteria with name, description, and weight",
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK,
        description="LLM backend to use for evaluations",
    )
    model_name: str | None = Field(default=None, description="Specific model to use")
    temperature: float = Field(default=0.1, description="LLM temperature for response generation")


def get_session_or_error(
    session_id: str,
    components: ServerComponents,
) -> tuple[DecisionSession | None, dict[str, Any] | None]:
    """Get session or return error dict for consistent session validation.

    Args:
        session_id: ID of the session to retrieve
        components: Server components container

    Returns:
        Tuple of (session, None) if successful, or (None, error_dict) if failed
    """
    # Use the new tuple-returning validation method
    session, error_dict = components.validation_service.validate_session_exists(
        session_id,
        components.session_manager,
    )

    if error_dict:
        # Return the simple error dict directly for handler tests
        return None, error_dict

    return session, None


async def _execute_parallel_evaluation(
    session: DecisionSession,
    components: ServerComponents,
) -> dict[str, dict[str, tuple[float | None, str]]]:
    """Execute parallel evaluation of options across criteria.

    Args:
        session: Decision session with options and criteria
        components: Server components containing decision service

    Returns:
        Evaluation results dict
    """
    return await components.decision_service.execute_parallel_evaluation(session)


@mcp.tool(
    description="When facing multiple options and need structured evaluation - create a decision matrix to systematically compare choices across weighted criteria",
)
async def start_decision_analysis(  # noqa: PLR0911
    topic: str,
    options: list[str],
    initial_criteria: list[dict[str, Any]] | None = None,
    model_backend: ModelBackend = ModelBackend.BEDROCK,
    model_name: str | None = None,
    temperature: float = 0.1,
    *,
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Initialize a new decision analysis session with options and optional criteria."""
    components = get_server_components()

    # Create request object from parameters
    request = StartDecisionAnalysisRequest(
        topic=topic,
        options=options,
        initial_criteria=initial_criteria,
        model_backend=model_backend,
        model_name=model_name,
        temperature=temperature,
    )

    # Validate topic
    if not components.validation_service.validate_topic(request.topic):
        return components.response_service.create_error_response(
            ERROR_MESSAGES["topic"],
            "Invalid topic",
        )

    # Validate options list
    if not request.options or len(request.options) < ValidationLimits.MIN_OPTIONS_REQUIRED:
        return components.response_service.create_error_response(
            "Need at least 2 options to create a meaningful decision matrix",
            "Invalid options",
        )
    if len(request.options) > ValidationLimits.MAX_OPTIONS_ALLOWED:
        return components.response_service.create_error_response(
            f"Too many options (max {ValidationLimits.MAX_OPTIONS_ALLOWED}). Consider grouping similar options.",
            "Invalid options",
        )
    # Validate individual option names
    for option_name in request.options:
        if not components.validation_service.validate_option_name(option_name):
            return components.response_service.create_error_response(
                ERROR_MESSAGES["option_name"],
                "Invalid option name",
            )

    try:
        # Create the session
        session = components.decision_service.create_session(
            topic=request.topic,
            initial_options=request.options,
            temperature=request.temperature,
        )

        # Validate and process initial criteria if provided
        if request.initial_criteria:
            validation_error = components.validation_service.validate_criteria_spec(
                request.initial_criteria,
            )
            if validation_error:
                return validation_error

        # Add criteria to session
        criteria_added = components.decision_service.process_initial_criteria(request, session)

        # Create standardized response
        return components.response_service.create_session_response(
            session,
            request,
            criteria_added,
        )

    except ValidationError as e:
        logger.warning("Invalid input for decision session: %s", e)
        return components.response_service.create_error_response(e.user_message, "Session creation")
    except ResourceLimitError as e:
        logger.warning("Resource limit exceeded: %s", e)
        return components.response_service.create_error_response(e.user_message, "Resource limit")
    except Exception:
        logger.exception("Unexpected error creating decision session")
        return components.response_service.create_error_response(
            "Failed to create session due to an unexpected error",
            "Unexpected error",
        )


class AddCriterionRequest(BaseModel):
    """Request to add an evaluation criterion."""

    session_id: str = Field(description="Session ID to add criterion to")
    name: str = Field(description="Name of the criterion (e.g., 'performance', 'cost')")
    description: str = Field(description="What this criterion evaluates")
    weight: float = Field(default=1.0, description="Importance weight (0.1-10.0, default 1.0)")
    custom_prompt: str | None = Field(
        default=None,
        description="Custom evaluation prompt for this criterion",
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK,
        description="LLM backend to use for this criterion",
    )
    model_name: str | None = Field(default=None, description="Specific model to use")
    temperature: float | None = Field(
        default=None,
        description="LLM temperature (None to use session default)",
    )


@mcp.tool(
    description="When you identify another factor to consider - add evaluation criteria with weights to structure your decision analysis",
)
async def add_criterion(  # noqa: PLR0911
    session_id: str,
    name: str,
    description: str,
    weight: float = 1.0,
    custom_prompt: str | None = None,
    model_backend: ModelBackend = ModelBackend.BEDROCK,
    model_name: str | None = None,
    temperature: float | None = None,
    *,
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Add a new evaluation criterion to an existing decision session."""
    components = get_server_components()

    # Create request object from parameters
    request = AddCriterionRequest(
        session_id=session_id,
        name=name,
        description=description,
        weight=weight,
        custom_prompt=custom_prompt,
        model_backend=model_backend,
        model_name=model_name,
        temperature=temperature,
    )

    # Get session and handle errors
    session, error = get_session_or_error(request.session_id, components)
    if error:
        return components.response_service.create_error_response(
            error["error"],
            "Session retrieval",
        )

    # Session validation guard
    assert (
        session is not None
    ), "Session should not be None after successful get_session_or_error"  # nosec B101

    # Validate criterion name
    if not components.validation_service.validate_criterion_name(request.name):
        return components.response_service.create_error_response(
            ERROR_MESSAGES["name"],
            "Invalid criterion name",
        )

    # Validate description
    if not components.validation_service.validate_description(request.description):
        return components.response_service.create_error_response(
            ERROR_MESSAGES["description"],
            "Invalid description",
        )

    # Validate weight
    if not components.validation_service.validate_weight(request.weight):
        return components.response_service.create_error_response(
            ERROR_MESSAGES["weight"],
            "Invalid weight",
        )

    # Check if criterion already exists
    if components.validation_service.validate_criterion_exists(session, request.name):
        return components.response_service.create_error_response(
            f"Criterion '{request.name}' already exists",
            "Duplicate criterion",
        )

    try:
        # Create and add criterion
        criterion = components.decision_service.create_criterion_from_request(request, session)
        components.decision_service.add_criterion_to_session(session, criterion)

        # Create standardized response
        return components.response_service.create_criterion_response(request, session)

    except SessionError as e:
        logger.warning("Session error when adding criterion: %s", e)
        return components.response_service.create_error_response(e.user_message, "Session error")
    except ValidationError as e:
        logger.warning("Invalid criterion input: %s", e)
        return components.response_service.create_error_response(e.user_message, "Validation error")
    except Exception:
        logger.exception("Unexpected error adding criterion")
        return components.response_service.create_error_response(
            "Failed to add criterion due to an unexpected error",
            "Unexpected error",
        )


class EvaluateOptionsRequest(BaseModel):
    """Request to evaluate all options against all criteria."""

    session_id: str = Field(description="Session ID for evaluation")


@mcp.tool(
    description="When ready to score your options systematically - run parallel evaluation where each criterion scores every option independently",
)
async def evaluate_options(
    session_id: str,
    *,
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Evaluate all options across all criteria using parallel thread orchestration."""
    components = get_server_components()

    # Create request object from parameters
    request = EvaluateOptionsRequest(session_id=session_id)

    # Get session and handle errors
    session, error = get_session_or_error(request.session_id, components)
    if error:
        return components.response_service.create_error_response(
            error["error"],
            "Session retrieval",
        )

    # Session validation guard
    assert (
        session is not None
    ), "Session should not be None after successful get_session_or_error"  # nosec B101

    # Validate prerequisites
    validation_error = components.validation_service.validate_evaluation_prerequisites(session)
    if validation_error:
        return components.response_service.create_error_response(
            validation_error["error"],
            "Prerequisites",
        )

    try:
        # Execute parallel evaluation
        evaluation_results = await _execute_parallel_evaluation(session, components)

        # Process results and create response
        total_scores, abstentions, errors = components.decision_service.process_evaluation_results(
            evaluation_results,
            session,
        )

        # Record evaluation in session history
        components.decision_service.record_evaluation(
            session,
            {
                "evaluation_results": evaluation_results,
                "total_scores": total_scores,
                "abstentions": abstentions,
                "errors": len(errors),
            },
        )

        # Create standardized response
        return components.response_service.create_evaluation_response(
            request.session_id,
            session,
            total_scores,
            abstentions,
            errors,
        )

    except Exception as e:
        logger.exception("Error during evaluation")
        return components.response_service.create_error_response(
            f"Evaluation failed: {e!s}",
            "Evaluation error",
        )


class GetDecisionMatrixRequest(BaseModel):
    """Request to get the complete decision matrix."""

    session_id: str = Field(description="Session ID to get matrix for")


@mcp.tool(
    description="When you need the complete picture - see the scored matrix with weighted totals, rankings, and clear recommendations",
)
async def get_decision_matrix(
    session_id: str,
    *,
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Get the complete decision matrix with scores, rankings, and recommendations."""
    components = get_server_components()

    # Create request object from parameters
    request = GetDecisionMatrixRequest(session_id=session_id)

    session, error = get_session_or_error(request.session_id, components)
    if error:
        return components.response_service.create_error_response(error["error"])

    # Session validation guard
    assert (
        session is not None
    ), "Session should not be None after successful get_session_or_error"  # nosec B101

    try:
        matrix_result = components.decision_service.get_decision_matrix(session)

        # Create standardized response
        return components.response_service.create_matrix_response(matrix_result, session)

    except (ValidationError, SessionError, ValueError, RuntimeError) as e:
        return components.error_handler.handle_exception(e, "Matrix generation")
    except Exception as e:  # noqa: BLE001
        return components.response_service.create_error_response(
            f"Failed to generate matrix: {e!s}",
            "Matrix generation",
        )


class AddOptionRequest(BaseModel):
    """Request to add a new option to existing analysis."""

    session_id: str = Field(description="Session ID to add option to")
    option_name: str = Field(description="Name of the new option")
    description: str | None = Field(default=None, description="Optional description")


@mcp.tool(
    description="When new alternatives emerge during analysis - add additional options to your existing decision matrix",
)
async def add_option(
    session_id: str,
    option_name: str,
    description: str | None = None,
    *,
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Add a new option to an existing decision analysis."""
    components = get_server_components()

    # Create request object from parameters
    request = AddOptionRequest(
        session_id=session_id,
        option_name=option_name,
        description=description,
    )

    # Validate option name
    if not components.validation_service.validate_option_name(request.option_name):
        return components.response_service.create_error_response(
            ERROR_MESSAGES["option_name"],
            "Invalid option name",
        )

    session, error = get_session_or_error(request.session_id, components)
    if error:
        return components.response_service.create_error_response(error["error"])

    # Session validation guard
    assert (
        session is not None
    ), "Session should not be None after successful get_session_or_error"  # nosec B101

    # Check if option already exists
    if components.validation_service.validate_option_exists(session, request.option_name):
        return components.response_service.create_error_response(
            f"Option '{request.option_name}' already exists",
            "Duplicate option",
        )

    try:
        components.decision_service.add_option_to_session(
            session,
            request.option_name,
            request.description,
        )

        # Create standardized response
        return components.response_service.create_option_added_response(request, session)

    except Exception as e:
        logger.exception("Error adding option")
        return components.response_service.create_error_response(
            f"Failed to add option: {e!s}",
            "Add option error",
        )


@mcp.tool(description="List all active decision analysis sessions")
async def list_sessions(*, ctx: Context | None = None) -> dict[str, Any]:  # noqa: ARG001
    """List all active decision analysis sessions."""
    components = get_server_components()

    try:
        active_sessions = components.decision_service.list_active_sessions()
        stats = components.decision_service.get_session_stats()

        # Create standardized response
        return components.response_service.create_sessions_list_response(active_sessions, stats)

    except Exception as e:
        logger.exception("Error listing sessions")
        return components.response_service.create_error_response(
            f"Failed to list sessions: {e!s}",
            "List sessions error",
        )


@mcp.tool(description="Clear all active decision analysis sessions")
async def clear_all_sessions(*, ctx: Context | None = None) -> dict[str, Any]:  # noqa: ARG001
    """Clear all active sessions from the session manager."""
    components = get_server_components()

    try:
        active_sessions = components.decision_service.list_active_sessions()
        cleared_count = 0

        for session_id in list(active_sessions.keys()):
            if components.decision_service.remove_session(session_id):
                cleared_count += 1

        return {
            "cleared": cleared_count,
            "message": f"Cleared {cleared_count} active sessions",
            "stats": components.decision_service.get_session_stats(),
        }

    except Exception as e:
        logger.exception("Error clearing sessions")
        return components.response_service.create_error_response(
            f"Failed to clear sessions: {e!s}",
            "Clear sessions error",
        )


@mcp.tool(
    description="Quick check of your most recent analysis session - see topic and status without remembering session ID",
)
async def current_session(*, ctx: Context | None = None) -> dict[str, Any]:  # noqa: ARG001
    """Get the most recently created active session without needing the session ID."""
    components = get_server_components()

    try:
        session = components.decision_service.get_current_session()

        # Create standardized response
        return components.response_service.create_current_session_response(session)

    except Exception as e:
        logger.exception("Error getting current session")
        return components.response_service.create_error_response(
            f"Failed to get current session: {e!s}",
            "Current session error",
        )


@mcp.tool(
    description="Test AWS Bedrock connectivity and configuration for debugging connection issues",
)
async def test_aws_bedrock_connection() -> dict[str, Any]:
    """Test Bedrock connectivity and return detailed diagnostics."""
    components = get_server_components()

    try:
        # Test connectivity using decision service
        test_result = await components.decision_service.test_bedrock_connection()

        # Create standardized response
        return components.response_service.create_bedrock_test_response(test_result)

    except Exception as e:
        logger.exception("Error testing Bedrock connection")
        return components.response_service.create_error_response(
            f"Test failed: {e!s}",
            "Bedrock test error",
        )


def main() -> None:
    """Run the Decision Matrix MCP server (stdio transport)."""
    try:
        logger.info("Starting Decision Matrix MCP server (stdio)...")
        logger.debug("Initializing FastMCP...")

        # Initialize server components at startup
        initialize_server_components()
        logger.debug("Server components initialized")

        # Test server creation
        logger.debug("MCP server created: %s", mcp)
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
            logger.exception("Server error")
            traceback.print_exc(file=sys.stderr)
            raise
    finally:
        # Always clean up resources on exit
        try:
            components = get_server_components()
            components.cleanup()
            logger.info("Server resources cleaned up")
        except Exception:
            logger.exception("Error during cleanup")


def http_main(host: str = "127.0.0.1", port: int = 8081) -> None:
    """Run the Decision Matrix MCP server (HTTP transport).

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 8081)
    """
    logger.info("Starting Decision Matrix MCP server (HTTP) on %s:%s", host, port)

    # Initialize server components
    try:
        initialize_server_components()
        logger.info("Server components initialized")
    except Exception:
        logger.exception("Failed to initialize server")
        sys.exit(1)

    # Import HTTP transport
    from .transports import create_http_app

    # Create HTTP app
    app = create_http_app()

    # Run with uvicorn
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logger.debug("Module started as main")

    # Check if HTTP mode requested
    import os

    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        port = int(os.environ.get("MCP_HTTP_PORT", "8081"))
        http_main(port=port)
    else:
        main()
