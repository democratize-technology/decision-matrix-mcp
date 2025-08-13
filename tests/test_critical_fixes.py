#!/usr/bin/env python3
"""
Test suite for critical technical debt fixes:
- TD-2024-001: Dependency injection for global state management
- TD-2024-002: Type safety with None checks for Optional types
- TD-2024-003: Session validation guards
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from mcp.server.fastmcp import Context
import pytest

from decision_matrix_mcp import ServerComponents, create_server_components
from decision_matrix_mcp.models import Criterion, DecisionSession, ModelBackend, Option, Score
from decision_matrix_mcp.orchestrator import DecisionOrchestrator
from decision_matrix_mcp.session_manager import SessionManager

# Mock context for all tests
mock_ctx = Mock(spec=Context)


class TestDependencyInjection:
    """Test dependency injection implementation (TD-2024-001)"""

    def test_server_components_creation(self):
        """Test that ServerComponents properly creates instances"""
        components = create_server_components()

        assert isinstance(components, ServerComponents)
        assert isinstance(components.orchestrator, DecisionOrchestrator)
        assert isinstance(components.session_manager, SessionManager)

    def test_server_components_with_custom_instances(self):
        """Test ServerComponents with custom injected instances"""
        custom_orchestrator = DecisionOrchestrator()
        custom_session_manager = SessionManager()

        components = ServerComponents(
            orchestrator=custom_orchestrator,
            session_manager=custom_session_manager,
        )

        assert components.orchestrator is custom_orchestrator
        assert components.session_manager is custom_session_manager

    def test_server_components_cleanup(self):
        """Test that cleanup properly clears resources"""
        components = create_server_components()

        # Create a session
        session = components.session_manager.create_session(
            topic="Test Topic",
            initial_options=["Option 1", "Option 2"],
        )

        # Verify session exists
        assert components.session_manager.get_session(session.session_id) is not None

        # Cleanup
        components.cleanup()

        # Verify session is cleared
        assert components.session_manager.get_session(session.session_id) is None

    def test_no_global_state_pollution(self):
        """Test that multiple component instances don't share state"""
        components1 = create_server_components()
        components2 = create_server_components()

        # Create session in first instance
        session1 = components1.session_manager.create_session(
            topic="Topic 1",
            initial_options=["A", "B"],
        )

        # Create session in second instance
        session2 = components2.session_manager.create_session(
            topic="Topic 2",
            initial_options=["X", "Y"],
        )

        # Verify isolation
        assert components1.session_manager.get_session(session1.session_id) is not None
        assert components1.session_manager.get_session(session2.session_id) is None

        assert components2.session_manager.get_session(session1.session_id) is None
        assert components2.session_manager.get_session(session2.session_id) is not None


class TestTypeSafety:
    """Test type safety fixes for Optional types (TD-2024-002)"""

    def test_option_weighted_total_with_none_score(self):
        """Test that weighted total handles None scores safely"""
        option = Option(name="Test Option")
        criteria = {
            "criterion1": Criterion(name="criterion1", description="Test criterion 1", weight=2.0),
            "criterion2": Criterion(name="criterion2", description="Test criterion 2", weight=3.0),
        }

        # Add scores with one None value
        option.scores["criterion1"] = Score(
            criterion_name="criterion1",
            option_name="Test Option",
            score=8.0,
            justification="Good",
        )
        option.scores["criterion2"] = Score(
            criterion_name="criterion2",
            option_name="Test Option",
            score=None,
            justification="Abstained",
        )

        # Should not crash and should only count non-None scores
        total = option.get_weighted_total(criteria)
        assert total == 8.0  # Only criterion1 counted: 8.0 * 2.0 / 2.0

    def test_option_weighted_total_all_none_scores(self):
        """Test weighted total when all scores are None"""
        option = Option(name="Test Option")
        criteria = {"criterion1": Criterion(name="criterion1", description="Test", weight=2.0)}

        # Add abstained score
        option.scores["criterion1"] = Score(
            criterion_name="criterion1",
            option_name="Test Option",
            score=None,
            justification="N/A",
        )

        # Should return 0.0 when no valid scores
        total = option.get_weighted_total(criteria)
        assert total == 0.0

    def test_score_breakdown_with_none_values(self):
        """Test score breakdown handles None values correctly"""
        option = Option(name="Test Option")
        criteria = {"criterion1": Criterion(name="criterion1", description="Test", weight=2.0)}

        option.scores["criterion1"] = Score(
            criterion_name="criterion1",
            option_name="Test Option",
            score=None,
            justification="Abstained",
        )

        breakdown = option.get_score_breakdown(criteria)
        assert len(breakdown) == 1
        assert breakdown[0]["raw_score"] is None
        assert breakdown[0]["weighted_score"] is None
        assert breakdown[0]["abstained"] is True

    def test_decision_matrix_with_none_scores(self):
        """Test decision matrix generation with None scores"""
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Test Decision",
        )

        # Add options and criteria
        session.add_option("Option A")
        session.add_criterion(Criterion(name="Cost", description="Cost analysis", weight=2.0))

        # Add score with None value
        session.options["Option A"].scores["Cost"] = Score(
            criterion_name="Cost",
            option_name="Option A",
            score=None,
            justification="Cannot evaluate",
        )

        # Generate matrix - should not crash
        matrix = session.get_decision_matrix()

        assert "matrix" in matrix
        assert matrix["matrix"]["Option A"]["Cost"]["raw_score"] is None
        assert matrix["matrix"]["Option A"]["Cost"]["weighted_score"] is None


class TestSessionValidationGuards:
    """Test session validation guards (TD-2024-003)"""

    @pytest.mark.asyncio()
    async def test_add_criterion_with_invalid_session(self):
        """Test that add_criterion properly handles invalid session"""
        from decision_matrix_mcp import AddCriterionRequest, add_criterion, get_server_components

        components = get_server_components()

        # Test with non-existent session
        request = AddCriterionRequest(
            session_id="invalid-session-id",
            name="Test Criterion",
            description="Test Description",
        )

        result = await add_criterion(request, mock_ctx)

        # Should return error, not crash
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_session_validation_guard_assertion(self):
        """Test that session validation guards are in place"""
        from decision_matrix_mcp import get_server_components, get_session_or_error

        components = get_server_components()

        # Create a valid session
        session = components.session_manager.create_session(
            topic="Test",
            initial_options=["A", "B"],
        )

        # Test valid session retrieval
        retrieved_session, error = get_session_or_error(session.session_id, components)
        assert retrieved_session is not None
        assert error is None
        assert retrieved_session.session_id == session.session_id

        # Test invalid session retrieval
        invalid_session, error = get_session_or_error("invalid-id", components)
        assert invalid_session is None
        assert error is not None
        assert "error" in error

    @pytest.mark.asyncio()
    async def test_all_handlers_have_session_guards(self):
        """Test that all session-dependent handlers have proper guards"""
        from decision_matrix_mcp import (
            AddCriterionRequest,
            AddOptionRequest,
            EvaluateOptionsRequest,
            GetDecisionMatrixRequest,
            add_criterion,
            add_option,
            evaluate_options,
            get_decision_matrix,
            get_server_components,
        )

        components = get_server_components()
        invalid_session_id = "non-existent-session"

        # Test add_criterion
        result = await add_criterion(
            AddCriterionRequest(session_id=invalid_session_id, name="Test", description="Test"),
            mock_ctx,
        )
        assert "error" in result

        # Test evaluate_options
        result = await evaluate_options(
            EvaluateOptionsRequest(session_id=invalid_session_id),
            mock_ctx,
        )
        assert "error" in result

        # Test get_decision_matrix
        result = await get_decision_matrix(
            GetDecisionMatrixRequest(session_id=invalid_session_id),
            mock_ctx,
        )
        assert "error" in result

        # Test add_option
        result = await add_option(
            AddOptionRequest(session_id=invalid_session_id, option_name="Test Option"),
            mock_ctx,
        )
        assert "error" in result


class TestIntegration:
    """Integration tests to ensure fixes work together"""

    @pytest.mark.asyncio()
    async def test_full_workflow_with_fixes(self):
        """Test complete decision analysis workflow with all fixes"""
        from decision_matrix_mcp import (
            AddCriterionRequest,
            EvaluateOptionsRequest,
            GetDecisionMatrixRequest,
            StartDecisionAnalysisRequest,
            add_criterion,
            evaluate_options,
            get_decision_matrix,
            get_server_components,
            start_decision_analysis,
        )

        # Use dependency injection
        components = get_server_components()

        # Start analysis
        start_result = await start_decision_analysis(
            StartDecisionAnalysisRequest(
                topic="Choose Framework",
                options=["React", "Vue", "Angular"],
                model_backend=ModelBackend.BEDROCK,
            ),
            mock_ctx,
        )

        assert "session_id" in start_result
        session_id = start_result["session_id"]

        # Add criterion
        criterion_result = await add_criterion(
            AddCriterionRequest(
                session_id=session_id,
                name="Performance",
                description="Runtime performance",
                weight=3.0,
            ),
            mock_ctx,
        )

        assert "error" not in criterion_result
        assert criterion_result["criterion_added"] == "Performance"

        # Mock orchestrator to avoid actual LLM calls
        with patch.object(components.orchestrator, "evaluate_options_across_criteria") as mock_eval:
            # Setup mock to return scores with some None values
            mock_eval.return_value = {
                "Performance": {
                    "React": (8.5, "Good performance"),
                    "Vue": (None, "Cannot evaluate"),
                    "Angular": (7.0, "Decent performance"),
                },
            }

            # Evaluate
            eval_result = await evaluate_options(
                EvaluateOptionsRequest(session_id=session_id),
                mock_ctx,
            )

            assert "error" not in eval_result
            assert eval_result["summary"]["successful_scores"] == 2  # Two non-None scores
            assert eval_result["summary"]["abstentions"] == 1  # One None score

        # Get matrix
        matrix_result = await get_decision_matrix(
            GetDecisionMatrixRequest(session_id=session_id),
            mock_ctx,
        )

        assert "error" not in matrix_result
        assert "rankings" in matrix_result
        assert len(matrix_result["rankings"]) == 3

        # Cleanup
        components.cleanup()

    def test_thread_safety_with_dependency_injection(self):
        """Test that dependency injection prevents thread safety issues"""
        import threading
        import time

        results = {"errors": [], "sessions": []}

        def create_sessions(thread_id):
            """Create sessions in parallel"""
            try:
                components = create_server_components()
                for i in range(5):
                    session = components.session_manager.create_session(
                        topic=f"Topic {thread_id}-{i}",
                        initial_options=["A", "B"],
                    )
                    results["sessions"].append(session.session_id)
                    time.sleep(0.01)  # Small delay to increase chance of race conditions
            except Exception as e:
                results["errors"].append(str(e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_sessions, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors and all sessions were created
        assert len(results["errors"]) == 0
        assert len(results["sessions"]) == 25  # 5 threads * 5 sessions each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
