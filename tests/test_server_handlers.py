"""Tests for MCP server handlers in __init__.py"""

from unittest.mock import patch

import pytest

from decision_matrix_mcp import (
    AddCriterionRequest,
    AddOptionRequest,
    EvaluateOptionsRequest,
    GetDecisionMatrixRequest,
    StartDecisionAnalysisRequest,
    add_criterion,
    add_option,
    clear_all_sessions,
    create_server_components,
    evaluate_options,
    get_decision_matrix,
    get_session_or_error,
    get_server_components,
    list_sessions,
    start_decision_analysis,
)
from decision_matrix_mcp.exceptions import ResourceLimitError, SessionError, ValidationError
from decision_matrix_mcp.models import Criterion, ModelBackend, Score


# Create server components for testing
server_components = create_server_components()
orchestrator = server_components.orchestrator
session_manager = server_components.session_manager


@pytest.fixture(autouse=True)
def patch_server_components(monkeypatch):
    """Automatically patch get_server_components to return our test components"""
    monkeypatch.setattr("decision_matrix_mcp.get_server_components", lambda: server_components)


class TestSessionHelpers:
    """Test helper functions for session management"""

    def test_get_session_or_error_invalid_id(self):
        """Test get_session_or_error with invalid session ID"""
        # Use empty string which is invalid
        session, error = get_session_or_error("", server_components)
        assert session is None
        assert error == {"error": "Invalid session ID format"}

    def test_get_session_or_error_not_found(self):
        """Test get_session_or_error with non-existent session"""
        valid_uuid = "12345678-1234-5678-1234-567812345678"
        session, error = get_session_or_error(valid_uuid, server_components)
        assert session is None
        assert error == {"error": f"Session {valid_uuid} not found or expired"}

    def test_get_session_or_error_success(self):
        """Test get_session_or_error with valid session"""
        # Create a session
        test_session = session_manager.create_session("Test topic", ["Option 1", "Option 2"])

        # Get it back
        session, error = get_session_or_error(test_session.session_id, server_components)
        assert session is not None
        assert error is None
        assert session.topic == "Test topic"

        # Cleanup
        session_manager.remove_session(test_session.session_id)


class TestStartDecisionAnalysis:
    """Test start_decision_analysis handler"""

    @pytest.mark.asyncio
    async def test_start_decision_analysis_success(self):
        """Test successful session creation"""
        request = StartDecisionAnalysisRequest(
            topic="Choose a database",
            options=["PostgreSQL", "MongoDB", "Redis"],
            model_backend=ModelBackend.BEDROCK,
        )

        result = await start_decision_analysis(request)

        assert "session_id" in result
        assert result["topic"] == "Choose a database"
        assert result["options"] == ["PostgreSQL", "MongoDB", "Redis"]
        assert result["model_backend"] == "bedrock"
        assert "Decision analysis initialized with 3 options" in result["message"]
        assert "next_steps" in result

        # Cleanup
        session_manager.remove_session(result["session_id"])

    @pytest.mark.asyncio
    async def test_start_decision_analysis_with_llm_params(self):
        """Test session creation with custom LLM parameters"""
        request = StartDecisionAnalysisRequest(
            topic="Choose a model",
            options=["GPT-4", "Claude", "Llama"],
            temperature=0.7,
            seed=42
        )
        
        result = await start_decision_analysis(request)
        
        assert "session_id" in result
        
        # Verify session has custom defaults
        session = session_manager.get_session(result["session_id"])
        assert session.default_temperature == 0.7
        assert session.default_seed == 42
        
        # Cleanup
        session_manager.remove_session(result["session_id"])

    @pytest.mark.asyncio
    async def test_start_decision_analysis_with_initial_criteria(self):
        """Test session creation with initial criteria"""
        request = StartDecisionAnalysisRequest(
            topic="Choose a framework",
            options=["React", "Vue", "Angular"],
            initial_criteria=[
                {"name": "Performance", "description": "Speed and efficiency", "weight": 2.0},
                {"name": "Learning Curve", "description": "Ease of learning", "weight": 1.5},
            ],
        )

        result = await start_decision_analysis(request)

        assert result["criteria_added"] == ["Performance", "Learning Curve"]
        assert "2 criteria" in result["message"]

        # Cleanup
        session_manager.remove_session(result["session_id"])

    @pytest.mark.asyncio
    async def test_start_decision_analysis_invalid_topic(self):
        """Test validation of topic"""
        # Empty topic
        request = StartDecisionAnalysisRequest(topic="", options=["A", "B"])
        result = await start_decision_analysis(request)
        assert result == {"error": "Invalid topic: must be a non-empty string under 500 characters"}

        # Too long topic
        request = StartDecisionAnalysisRequest(topic="x" * 501, options=["A", "B"])
        result = await start_decision_analysis(request)
        assert result == {"error": "Invalid topic: must be a non-empty string under 500 characters"}

    @pytest.mark.asyncio
    async def test_start_decision_analysis_invalid_options(self):
        """Test validation of options"""
        # Too few options
        request = StartDecisionAnalysisRequest(topic="Test", options=["Only one"])
        result = await start_decision_analysis(request)
        assert result == {"error": "Need at least 2 options to create a meaningful decision matrix"}

        # Too many options
        request = StartDecisionAnalysisRequest(
            topic="Test",
            options=[f"Option{i}" for i in range(21)],
        )
        result = await start_decision_analysis(request)
        assert result == {"error": "Too many options (max 20). Consider grouping similar options."}

        # Invalid option name
        request = StartDecisionAnalysisRequest(topic="Test", options=["Valid", ""])
        result = await start_decision_analysis(request)
        assert "Invalid option name" in result["error"]

    @pytest.mark.asyncio
    async def test_start_decision_analysis_invalid_initial_criteria(self):
        """Test validation of initial criteria"""
        # Invalid criterion name
        request = StartDecisionAnalysisRequest(
            topic="Test",
            options=["A", "B"],
            initial_criteria=[{"name": "", "description": "Test", "weight": 1.0}],
        )
        result = await start_decision_analysis(request)
        assert "Invalid criterion name" in result["error"]

        # Invalid description
        request = StartDecisionAnalysisRequest(
            topic="Test",
            options=["A", "B"],
            initial_criteria=[{"name": "Cost", "description": "", "weight": 1.0}],
        )
        result = await start_decision_analysis(request)
        assert "Invalid criterion description" in result["error"]

        # Invalid weight
        request = StartDecisionAnalysisRequest(
            topic="Test",
            options=["A", "B"],
            initial_criteria=[{"name": "Cost", "description": "Test", "weight": 15.0}],
        )
        result = await start_decision_analysis(request)
        assert "Invalid weight" in result["error"]

    @pytest.mark.asyncio
    async def test_start_decision_analysis_validation_error(self):
        """Test handling ValidationError from session manager"""
        with patch.object(
            session_manager,
            "create_session",
            side_effect=ValidationError("Invalid input", "Please check your input"),
        ):
            request = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
            result = await start_decision_analysis(request)
            assert result == {"error": "Please check your input"}

    @pytest.mark.asyncio
    async def test_start_decision_analysis_resource_limit(self):
        """Test handling ResourceLimitError"""
        with patch.object(
            session_manager,
            "create_session",
            side_effect=ResourceLimitError("Too many sessions", "Session limit reached"),
        ):
            request = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
            result = await start_decision_analysis(request)
            assert result == {"error": "Session limit reached"}

    @pytest.mark.asyncio
    async def test_start_decision_analysis_unexpected_error(self):
        """Test handling unexpected errors"""
        with patch.object(
            session_manager,
            "create_session",
            side_effect=Exception("Unexpected error"),
        ):
            request = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
            result = await start_decision_analysis(request)
            assert result == {"error": "Failed to create session due to an unexpected error"}


class TestAddCriterion:
    """Test add_criterion handler"""

    @pytest.fixture
    def test_session(self):
        """Create a test session"""
        session = session_manager.create_session("Test", ["Option A", "Option B"])
        yield session
        session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio
    async def test_add_criterion_success(self, test_session):
        """Test successful criterion addition"""
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Performance",
            description="Evaluate performance characteristics",
            weight=2.5,
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",
        )

        result = await add_criterion(request)

        assert result["criterion_added"] == "Performance"
        assert result["weight"] == 2.5
        assert result["total_criteria"] == 1
        assert "Performance" in result["all_criteria"]

    @pytest.mark.asyncio
    async def test_add_criterion_with_custom_prompt(self, test_session):
        """Test adding criterion with custom prompt"""
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Security",
            description="Evaluate security aspects",
            custom_prompt="You are a security expert. Focus on vulnerabilities.",
        )

        result = await add_criterion(request)

        assert result["criterion_added"] == "Security"

        # Verify custom prompt was set
        session = session_manager.get_session(test_session.session_id)
        criterion = session.criteria["Security"]
        assert "security expert" in criterion.system_prompt

    @pytest.mark.asyncio
    async def test_add_criterion_with_llm_params(self, test_session):
        """Test adding criterion with custom LLM parameters"""
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Reliability",
            description="Evaluate reliability and uptime",
            temperature=0.3,
            seed=54321
        )
        
        result = await add_criterion(request)
        assert result["criterion_added"] == "Reliability"
        
        # Verify criterion has custom parameters
        criterion = test_session.criteria["Reliability"]
        assert criterion.temperature == 0.3
        assert criterion.seed == 54321

    @pytest.mark.asyncio
    async def test_add_criterion_inherits_session_defaults(self):
        """Test criterion inherits session defaults when not specified"""
        # Create session with custom defaults
        test_session = session_manager.create_session(
            "Test", ["A", "B"], temperature=0.8, seed=99999
        )
        
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="TestCriterion",
            description="Test description"
            # No temperature or seed specified
        )
        
        result = await add_criterion(request)
        
        # Verify criterion inherited session defaults
        criterion = test_session.criteria["TestCriterion"]
        assert criterion.temperature == 0.8
        assert criterion.seed == 99999
        
        # Cleanup
        session_manager.remove_session(test_session.session_id)

    @pytest.mark.asyncio
    async def test_add_criterion_invalid_session_id(self):
        """Test invalid session ID"""
        request = AddCriterionRequest(
            session_id="",  # Empty string is invalid
            name="Test",
            description="Test",
        )

        result = await add_criterion(request)
        assert result == {"error": "Invalid session ID"}

    @pytest.mark.asyncio
    async def test_add_criterion_session_not_found(self):
        """Test non-existent session"""
        request = AddCriterionRequest(
            session_id="12345678-1234-5678-1234-567812345678",
            name="Test",
            description="Test",
        )

        result = await add_criterion(request)
        assert "not found or expired" in result["error"]

    @pytest.mark.asyncio
    async def test_add_criterion_duplicate(self, test_session):
        """Test adding duplicate criterion"""
        # Add first criterion
        request1 = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Cost",
            description="Evaluate cost",
        )
        await add_criterion(request1)

        # Try to add same criterion again
        request2 = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Cost",
            description="Different description",
        )
        result = await add_criterion(request2)

        assert result == {"error": "Criterion 'Cost' already exists"}

    @pytest.mark.asyncio
    async def test_add_criterion_validation_errors(self, test_session):
        """Test input validation"""
        # Invalid name
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="x" * 101,
            description="Test",
        )
        result = await add_criterion(request)
        assert "Invalid criterion name" in result["error"]

        # Invalid description
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Test",
            description="x" * 1001,
        )
        result = await add_criterion(request)
        assert "Invalid description" in result["error"]

        # Invalid weight
        request = AddCriterionRequest(
            session_id=test_session.session_id,
            name="Test",
            description="Test",
            weight=0.05,
        )
        result = await add_criterion(request)
        assert "Invalid weight" in result["error"]

    @pytest.mark.asyncio
    async def test_add_criterion_session_error(self, test_session):
        """Test handling SessionError"""
        with patch.object(
            test_session,
            "add_criterion",
            side_effect=SessionError("Session error", "Cannot add criterion"),
        ):
            request = AddCriterionRequest(
                session_id=test_session.session_id,
                name="Test",
                description="Test",
            )

            with patch.object(session_manager, "get_session", return_value=test_session):
                result = await add_criterion(request)
                assert result == {"error": "Cannot add criterion"}

    @pytest.mark.asyncio
    async def test_add_criterion_unexpected_error(self, test_session):
        """Test handling unexpected errors"""
        with patch.object(test_session, "add_criterion", side_effect=Exception("Unexpected")):
            request = AddCriterionRequest(
                session_id=test_session.session_id,
                name="Test",
                description="Test",
            )

            with patch.object(session_manager, "get_session", return_value=test_session):
                result = await add_criterion(request)
                assert result == {"error": "Failed to add criterion due to an unexpected error"}


class TestEvaluateOptions:
    """Test evaluate_options handler"""

    @pytest.fixture
    def test_session_with_criteria(self):
        """Create a test session with options and criteria"""
        session = session_manager.create_session("Test", ["Option A", "Option B"])

        # Add criteria
        criterion1 = Criterion(name="Performance", description="Speed", weight=2.0)
        criterion2 = Criterion(name="Cost", description="Price", weight=1.5)
        session.add_criterion(criterion1)
        session.add_criterion(criterion2)

        yield session
        session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio
    async def test_evaluate_options_success(self, test_session_with_criteria):
        """Test successful evaluation"""
        mock_results = {
            "Performance": {
                "Option A": (8.0, "Good performance"),
                "Option B": (6.0, "Average performance"),
            },
            "Cost": {
                "Option A": (7.0, "Reasonable cost"),
                "Option B": (9.0, "Very cost effective"),
            },
        }

        with patch.object(
            orchestrator,
            "evaluate_options_across_criteria",
            return_value=mock_results,
        ):
            request = EvaluateOptionsRequest(session_id=test_session_with_criteria.session_id)
            result = await evaluate_options(request)

            assert result["evaluation_complete"] is True
            assert result["summary"]["options_evaluated"] == 2
            assert result["summary"]["criteria_used"] == 2
            assert result["summary"]["successful_scores"] == 4
            assert result["summary"]["abstentions"] == 0
            assert result["summary"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_options_with_abstentions(self, test_session_with_criteria):
        """Test evaluation with abstentions"""
        mock_results = {
            "Performance": {
                "Option A": (8.0, "Good performance"),
                "Option B": (None, "Not applicable"),
            },
            "Cost": {
                "Option A": (7.0, "Reasonable cost"),
                "Option B": (9.0, "Very cost effective"),
            },
        }

        with patch.object(
            orchestrator,
            "evaluate_options_across_criteria",
            return_value=mock_results,
        ):
            request = EvaluateOptionsRequest(session_id=test_session_with_criteria.session_id)
            result = await evaluate_options(request)

            assert result["summary"]["successful_scores"] == 3
            assert result["summary"]["abstentions"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_options_with_errors(self, test_session_with_criteria):
        """Test evaluation with errors"""
        mock_results = {
            "Performance": {
                "Option A": (8.0, "Good performance"),
                "Option B": (None, "Error: API timeout"),
            },
            "Cost": {
                "Option A": (None, "Error: Rate limit exceeded"),
                "Option B": (9.0, "Very cost effective"),
            },
        }

        with patch.object(
            orchestrator,
            "evaluate_options_across_criteria",
            return_value=mock_results,
        ):
            request = EvaluateOptionsRequest(session_id=test_session_with_criteria.session_id)
            result = await evaluate_options(request)

            assert result["summary"]["successful_scores"] == 2
            assert result["summary"]["errors"] == 2
            assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_options_invalid_session(self):
        """Test evaluation with invalid session ID"""
        request = EvaluateOptionsRequest(session_id="")
        result = await evaluate_options(request)
        assert result == {"error": "Invalid session ID"}

    @pytest.mark.asyncio
    async def test_evaluate_options_no_options(self):
        """Test evaluation with no options"""
        session = session_manager.create_session("Test", [])

        request = EvaluateOptionsRequest(session_id=session.session_id)
        result = await evaluate_options(request)
        assert result == {"error": "No options to evaluate. Add options first."}

        session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio
    async def test_evaluate_options_no_criteria(self):
        """Test evaluation with no criteria"""
        session = session_manager.create_session("Test", ["Option A", "Option B"])

        request = EvaluateOptionsRequest(session_id=session.session_id)
        result = await evaluate_options(request)
        assert result == {"error": "No criteria defined. Add criteria first."}

        session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio
    async def test_evaluate_options_orchestrator_error(self, test_session_with_criteria):
        """Test handling orchestrator errors"""
        with patch.object(
            orchestrator,
            "evaluate_options_across_criteria",
            side_effect=Exception("Orchestrator failed"),
        ):
            request = EvaluateOptionsRequest(session_id=test_session_with_criteria.session_id)
            result = await evaluate_options(request)
            assert result == {"error": "Evaluation failed: Orchestrator failed"}


class TestGetDecisionMatrix:
    """Test get_decision_matrix handler"""

    @pytest.fixture
    def evaluated_session(self):
        """Create a session with evaluation results"""
        session = session_manager.create_session("Test", ["Option A", "Option B"])

        # Add criteria
        criterion = Criterion(name="Performance", description="Speed", weight=2.0)
        session.add_criterion(criterion)

        # Add scores
        score_a = Score(
            criterion_name="Performance",
            option_name="Option A",
            score=8.0,
            justification="Good performance",
        )
        score_b = Score(
            criterion_name="Performance",
            option_name="Option B",
            score=6.0,
            justification="Average performance",
        )
        session.options["Option A"].add_score(score_a)
        session.options["Option B"].add_score(score_b)

        yield session
        session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio
    async def test_get_decision_matrix_success(self, evaluated_session):
        """Test successful matrix retrieval"""
        request = GetDecisionMatrixRequest(session_id=evaluated_session.session_id)
        result = await get_decision_matrix(request)

        assert "matrix" in result
        assert "rankings" in result
        assert "session_info" in result
        assert result["session_info"]["total_options"] == 2
        assert result["session_info"]["total_criteria"] == 1

    @pytest.mark.asyncio
    async def test_get_decision_matrix_invalid_session(self):
        """Test matrix retrieval with invalid session"""
        request = GetDecisionMatrixRequest(session_id="")
        result = await get_decision_matrix(request)
        assert result == {"error": "Invalid session ID"}

    @pytest.mark.asyncio
    async def test_get_decision_matrix_error_in_generation(self, evaluated_session):
        """Test handling errors in matrix generation"""
        with patch.object(
            evaluated_session,
            "get_decision_matrix",
            return_value={"error": "No evaluations run yet"},
        ):
            request = GetDecisionMatrixRequest(session_id=evaluated_session.session_id)

            with patch.object(session_manager, "get_session", return_value=evaluated_session):
                result = await get_decision_matrix(request)
                assert result == {"error": "No evaluations run yet"}

    @pytest.mark.asyncio
    async def test_get_decision_matrix_unexpected_error(self, evaluated_session):
        """Test handling unexpected errors"""
        with patch.object(
            evaluated_session,
            "get_decision_matrix",
            side_effect=Exception("Matrix generation failed"),
        ):
            request = GetDecisionMatrixRequest(session_id=evaluated_session.session_id)

            with patch.object(session_manager, "get_session", return_value=evaluated_session):
                result = await get_decision_matrix(request)
                assert result == {"error": "Failed to generate matrix: Matrix generation failed"}


class TestAddOption:
    """Test add_option handler"""

    @pytest.fixture
    def test_session(self):
        """Create a test session"""
        session = session_manager.create_session("Test", ["Option A"])
        yield session
        session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio
    async def test_add_option_success(self, test_session):
        """Test successful option addition"""
        request = AddOptionRequest(
            session_id=test_session.session_id,
            option_name="Option B",
            description="Second option",
        )

        result = await add_option(request)

        assert result["option_added"] == "Option B"
        assert result["total_options"] == 2
        assert "Option B" in result["all_options"]

    @pytest.mark.asyncio
    async def test_add_option_invalid_session(self):
        """Test adding option to invalid session"""
        request = AddOptionRequest(
            session_id="",  # Empty string is invalid
            option_name="Test",
        )

        result = await add_option(request)
        assert result == {"error": "Invalid session ID"}

    @pytest.mark.asyncio
    async def test_add_option_invalid_name(self, test_session):
        """Test validation of option name"""
        request = AddOptionRequest(
            session_id=test_session.session_id,
            option_name="",
        )

        result = await add_option(request)
        assert "Invalid option name" in result["error"]

    @pytest.mark.asyncio
    async def test_add_option_duplicate(self, test_session):
        """Test adding duplicate option"""
        request = AddOptionRequest(
            session_id=test_session.session_id,
            option_name="Option A",  # Already exists
        )

        result = await add_option(request)
        assert result == {"error": "Option 'Option A' already exists"}

    @pytest.mark.asyncio
    async def test_add_option_error_handling(self, test_session):
        """Test error handling"""
        with patch.object(test_session, "add_option", side_effect=Exception("Add failed")):
            request = AddOptionRequest(
                session_id=test_session.session_id,
                option_name="New Option",
            )

            with patch.object(session_manager, "get_session", return_value=test_session):
                result = await add_option(request)
                assert result == {"error": "Failed to add option: Add failed"}


class TestListSessions:
    """Test list_sessions handler"""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        """Test listing when no sessions exist"""
        # Clear any existing sessions
        for sid in list(session_manager.list_active_sessions().keys()):
            session_manager.remove_session(sid)

        result = await list_sessions()

        assert result["sessions"] == []
        assert result["total_active"] == 0
        assert "stats" in result

    @pytest.mark.asyncio
    async def test_list_sessions_with_sessions(self):
        """Test listing active sessions"""
        # Create test sessions
        session1 = session_manager.create_session("Database Selection", ["MySQL", "PostgreSQL"])

        session2 = session_manager.create_session("Frontend Framework", ["React", "Vue"])
        criterion = Criterion(name="Performance", description="Speed", weight=2.0)
        session2.add_criterion(criterion)

        # Add evaluation to session2
        session2.record_evaluation({"test": "data"})

        result = await list_sessions()

        assert result["total_active"] >= 2

        # Find our test sessions
        sessions_by_topic = {s["topic"]: s for s in result["sessions"]}

        assert "Database Selection" in sessions_by_topic
        assert sessions_by_topic["Database Selection"]["status"] == "setup"
        assert sessions_by_topic["Database Selection"]["options"] == ["MySQL", "PostgreSQL"]

        assert "Frontend Framework" in sessions_by_topic
        assert sessions_by_topic["Frontend Framework"]["status"] == "evaluated"
        assert sessions_by_topic["Frontend Framework"]["criteria"] == ["Performance"]

        # Cleanup
        session_manager.remove_session(session1.session_id)
        session_manager.remove_session(session2.session_id)

    @pytest.mark.asyncio
    async def test_list_sessions_error_handling(self):
        """Test error handling in list_sessions"""
        with patch.object(
            session_manager, "list_active_sessions", side_effect=Exception("List failed")
        ):
            result = await list_sessions()
            assert result == {"error": "Failed to list sessions: List failed"}


class TestClearAllSessions:
    """Test clear_all_sessions handler"""

    @pytest.mark.asyncio
    async def test_clear_all_sessions_success(self):
        """Test clearing all sessions"""
        # Create test sessions
        session1 = session_manager.create_session("Test1", ["A", "B"])
        session2 = session_manager.create_session("Test2", ["C", "D"])

        result = await clear_all_sessions()

        assert result["cleared"] >= 2
        assert "Cleared" in result["message"]
        assert "stats" in result

        # Verify sessions are gone
        assert session_manager.get_session(session1.session_id) is None
        assert session_manager.get_session(session2.session_id) is None

    @pytest.mark.asyncio
    async def test_clear_all_sessions_empty(self):
        """Test clearing when no sessions exist"""
        # Clear any existing sessions first
        for sid in list(session_manager.list_active_sessions().keys()):
            session_manager.remove_session(sid)

        result = await clear_all_sessions()

        assert result["cleared"] == 0
        assert "Cleared 0 active sessions" in result["message"]

    @pytest.mark.asyncio
    async def test_clear_all_sessions_error_handling(self):
        """Test error handling"""
        with patch.object(
            session_manager, "list_active_sessions", side_effect=Exception("Clear failed")
        ):
            result = await clear_all_sessions()
            assert result == {"error": "Failed to clear sessions: Clear failed"}
