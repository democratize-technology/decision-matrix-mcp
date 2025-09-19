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

"""Decision Service - Core business logic for decision analysis.

Handles session lifecycle, option/criterion management, and evaluation orchestration.
"""

import logging
from typing import Any

from ..models import Criterion, DecisionSession, Score
from ..orchestrator import DecisionOrchestrator
from ..session_manager import SessionManager

logger = logging.getLogger(__name__)


class DecisionService:
    """Core business logic service for decision analysis operations."""

    def __init__(self, session_manager: SessionManager, orchestrator: DecisionOrchestrator) -> None:
        """Initialize DecisionService with dependencies.

        Args:
            session_manager: Session management component
            orchestrator: Evaluation orchestration component
        """
        self.session_manager = session_manager
        self.orchestrator = orchestrator

    def create_session(
        self,
        topic: str,
        initial_options: list[str],
        temperature: float = 0.1,
    ) -> DecisionSession:
        """Create a new decision analysis session.

        Args:
            topic: The decision topic or question
            initial_options: List of options to evaluate
            temperature: LLM temperature for evaluation

        Returns:
            Created DecisionSession
        """
        return self.session_manager.create_session(
            topic=topic,
            initial_options=initial_options,
            temperature=temperature,
        )

    def get_session(self, session_id: str) -> DecisionSession | None:
        """Retrieve a session by ID.

        Args:
            session_id: UUID of the session

        Returns:
            DecisionSession if found, None otherwise
        """
        return self.session_manager.get_session(session_id)

    def create_criterion_from_request(self, request: Any, session: DecisionSession) -> Criterion:
        """Create a Criterion object from request data.

        Args:
            request: Request containing criterion parameters
            session: Decision session for default temperature

        Returns:
            Criterion object
        """
        criterion = Criterion(
            name=request.name,
            description=request.description,
            weight=request.weight,
            model_backend=request.model_backend,
            model_name=request.model_name,
            temperature=(
                request.temperature
                if request.temperature is not None
                else session.default_temperature
            ),
        )

        # Override system prompt if provided
        if hasattr(request, "custom_prompt") and request.custom_prompt:
            criterion.system_prompt = request.custom_prompt

        return criterion

    def add_criterion_to_session(self, session: DecisionSession, criterion: Criterion) -> None:
        """Add a criterion to a session.

        Args:
            session: Decision session to modify
            criterion: Criterion to add

        Raises:
            SessionError: If criterion already exists or session is invalid
        """
        session.add_criterion(criterion)

    def add_option_to_session(
        self,
        session: DecisionSession,
        option_name: str,
        description: str | None = None,
    ) -> None:
        """Add an option to a session.

        Args:
            session: Decision session to modify
            option_name: Name of the option
            description: Optional description

        Raises:
            SessionError: If option already exists or session is invalid
        """
        session.add_option(option_name, description)

    def process_initial_criteria(self, request: Any, session: DecisionSession) -> list[str]:
        """Process and add initial criteria to a session.

        Args:
            request: Request containing initial_criteria and model settings
            session: Decision session to add criteria to

        Returns:
            List of criteria names that were added
        """
        criteria_added: list[str] = []

        if not hasattr(request, "initial_criteria") or not request.initial_criteria:
            return criteria_added

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
                temperature=criterion_spec.get("temperature", request.temperature),
            )

            session.add_criterion(criterion)
            criteria_added.append(name)

        return criteria_added

    async def execute_parallel_evaluation(
        self,
        session: DecisionSession,
    ) -> dict[str, dict[str, tuple[float | None, str]]]:
        """Execute parallel evaluation of options across criteria.

        Args:
            session: Decision session with options and criteria

        Returns:
            Evaluation results dict mapping criterion->option->score_tuple
        """
        logger.info(
            "Starting evaluation: %d options x %d criteria",
            len(session.options),
            len(session.criteria),
        )

        return await self.orchestrator.evaluate_options_across_criteria(
            session.threads,
            list(session.options.values()),
        )

    def process_evaluation_results(
        self,
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

    def record_evaluation(self, session: DecisionSession, evaluation_data: dict[str, Any]) -> None:
        """Record evaluation results in session history.

        Args:
            session: Decision session to update
            evaluation_data: Evaluation metadata and results
        """
        session.record_evaluation(evaluation_data)

    def get_decision_matrix(self, session: DecisionSession) -> dict[str, Any]:
        """Get the complete decision matrix for a session.

        Args:
            session: Decision session to analyze

        Returns:
            Decision matrix results with scores and rankings

        Raises:
            ValidationError: If session is not ready for matrix generation
        """
        return session.get_decision_matrix()

    def list_active_sessions(self) -> dict[str, DecisionSession]:
        """List all active decision analysis sessions.

        Returns:
            Dictionary mapping session_id to DecisionSession
        """
        return self.session_manager.list_active_sessions()

    def get_current_session(self) -> DecisionSession | None:
        """Get the most recently created active session.

        Returns:
            Most recent DecisionSession or None if no active sessions
        """
        return self.session_manager.get_current_session()

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from the manager.

        Args:
            session_id: UUID of session to remove

        Returns:
            True if session was removed, False if not found
        """
        return self.session_manager.remove_session(session_id)

    def get_session_stats(self) -> dict[str, Any]:
        """Get session manager statistics.

        Returns:
            Dictionary with session statistics
        """
        return self.session_manager.get_stats()

    async def test_bedrock_connection(self) -> dict[str, Any]:
        """Test AWS Bedrock connectivity.

        Returns:
            Test result with status and diagnostics
        """
        return await self.orchestrator.test_bedrock_connection()
