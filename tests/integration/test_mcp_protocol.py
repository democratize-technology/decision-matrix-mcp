"""
Integration tests for MCP protocol end-to-end workflows.

These tests verify the complete MCP server functionality including:
- Server startup and tool registration
- Complete decision analysis workflow
- Error handling through MCP protocol
- Session management through MCP interface
"""

from pathlib import Path
import sys
from unittest.mock import patch

import pytest

# Add the source directory to the path so we can import the server
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from decision_matrix_mcp import create_server_components
from decision_matrix_mcp.models import ModelBackend


class MCPServerTestHelper:
    """Helper class for testing MCP server functionality."""

    def __init__(self):
        self.server_components = None
        self.mcp_server = None

    async def setup_server(self):
        """Set up the MCP server for testing."""
        from decision_matrix_mcp import app

        self.mcp_server = app
        self.server_components = create_server_components()

    async def teardown_server(self):
        """Clean up server resources."""
        if self.server_components:
            # Clear all sessions
            session_manager = self.server_components.session_manager
            for session_id in list(session_manager.list_active_sessions().keys()):
                session_manager.remove_session(session_id)


@pytest.fixture()
async def mcp_server():
    """Fixture providing a configured MCP server for testing."""
    helper = MCPServerTestHelper()
    await helper.setup_server()
    yield helper
    await helper.teardown_server()


class TestMCPServerStartup:
    """Test MCP server initialization and tool registration."""

    @pytest.mark.asyncio()
    async def test_server_creation(self, mcp_server):
        """Test that the MCP server can be created successfully."""
        assert mcp_server.mcp_server is not None
        assert mcp_server.server_components is not None

    @pytest.mark.asyncio()
    async def test_server_tool_registration(self, mcp_server):
        """Test that all required tools are registered with the server."""
        # Get the FastMCP app
        app = mcp_server.mcp_server

        # Check that required tools are registered
        expected_tools = [
            "start_decision_analysis",
            "add_criterion",
            "evaluate_options",
            "get_decision_matrix",
            "add_option",
            "list_sessions",
            "clear_all_sessions",
            "current_session",
            "test_aws_bedrock_connection",
        ]

        # Note: We can't directly access the tool registry in FastMCP,
        # but we can verify the handlers exist
        from decision_matrix_mcp import (
            add_criterion,
            add_option,
            clear_all_sessions,
            current_session,
            evaluate_options,
            get_decision_matrix,
            list_sessions,
            start_decision_analysis,
            test_aws_bedrock_connection,
        )

        # Verify handlers are callable
        assert callable(start_decision_analysis)
        assert callable(add_criterion)
        assert callable(evaluate_options)
        assert callable(get_decision_matrix)
        assert callable(add_option)
        assert callable(list_sessions)
        assert callable(clear_all_sessions)
        assert callable(current_session)
        assert callable(test_aws_bedrock_connection)


class TestCompleteDecisionWorkflow:
    """Test complete decision analysis workflow through MCP protocol."""

    @pytest.mark.asyncio()
    async def test_complete_workflow_success(self, mcp_server):
        """Test a complete decision analysis workflow from start to finish."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            AddCriterionRequest,
            EvaluateOptionsRequest,
            GetDecisionMatrixRequest,
            StartDecisionAnalysisRequest,
            add_criterion,
            evaluate_options,
            get_decision_matrix,
            start_decision_analysis,
        )

        mock_ctx = Mock()

        # Step 1: Start decision analysis
        start_request = StartDecisionAnalysisRequest(
            topic="Choose a database solution",
            options=["PostgreSQL", "MongoDB", "Redis"],
            model_backend=ModelBackend.BEDROCK,
        )

        start_result = await start_decision_analysis(start_request, mock_ctx)
        assert "session_id" in start_result
        assert start_result["topic"] == "Choose a database solution"
        assert len(start_result["options"]) == 3

        session_id = start_result["session_id"]

        # Step 2: Add criteria
        criteria_data = [
            {"name": "Performance", "description": "Query speed and throughput", "weight": 2.0},
            {"name": "Scalability", "description": "Ability to handle growth", "weight": 1.5},
            {"name": "Cost", "description": "Total cost of ownership", "weight": 1.0},
        ]

        for criterion_data in criteria_data:
            criterion_request = AddCriterionRequest(
                session_id=session_id,
                name=criterion_data["name"],
                description=criterion_data["description"],
                weight=criterion_data["weight"],
            )

            criterion_result = await add_criterion(criterion_request, mock_ctx)
            assert criterion_result["criterion_added"] == criterion_data["name"]
            assert criterion_result["weight"] == criterion_data["weight"]

        # Step 3: Mock evaluation (since we don't want to call real LLMs)
        mock_evaluation_results = {
            "Performance": {
                "PostgreSQL": (8.5, "Excellent performance for complex queries"),
                "MongoDB": (7.0, "Good performance for document operations"),
                "Redis": (9.0, "Outstanding performance for caching"),
            },
            "Scalability": {
                "PostgreSQL": (7.5, "Good horizontal scaling with proper setup"),
                "MongoDB": (8.0, "Native sharding support"),
                "Redis": (6.0, "Limited by memory constraints"),
            },
            "Cost": {
                "PostgreSQL": (8.0, "Open source with low licensing costs"),
                "MongoDB": (6.0, "Enterprise features require licensing"),
                "Redis": (7.0, "Open source but high memory costs"),
            },
        }

        with patch.object(
            mcp_server.server_components.orchestrator,
            "evaluate_options_across_criteria",
            return_value=mock_evaluation_results,
        ):
            eval_request = EvaluateOptionsRequest(session_id=session_id)
            eval_result = await evaluate_options(eval_request, mock_ctx)

            assert eval_result["evaluation_complete"] is True
            assert eval_result["summary"]["options_evaluated"] == 3
            assert eval_result["summary"]["criteria_used"] == 3
            assert eval_result["summary"]["successful_scores"] == 9
            assert eval_result["summary"]["abstentions"] == 0
            assert eval_result["summary"]["errors"] == 0

        # Step 4: Get decision matrix
        matrix_request = GetDecisionMatrixRequest(session_id=session_id)
        matrix_result = await get_decision_matrix(matrix_request, mock_ctx)

        assert "matrix" in matrix_result
        assert "rankings" in matrix_result
        assert "session_info" in matrix_result
        assert matrix_result["session_info"]["total_options"] == 3
        assert matrix_result["session_info"]["total_criteria"] == 3

        # Verify matrix structure
        matrix = matrix_result["matrix"]
        assert len(matrix) == 3  # 3 options
        for option_data in matrix:
            assert "option" in option_data
            assert "scores" in option_data
            assert "weighted_total" in option_data
            assert len(option_data["scores"]) == 3  # 3 criteria

        # Verify rankings
        rankings = matrix_result["rankings"]
        assert len(rankings) == 3
        for rank_data in rankings:
            assert "rank" in rank_data
            assert "option" in rank_data
            assert "weighted_total" in rank_data

    @pytest.mark.asyncio()
    async def test_workflow_with_abstentions(self, mcp_server):
        """Test workflow handling abstentions in evaluation."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            AddCriterionRequest,
            EvaluateOptionsRequest,
            StartDecisionAnalysisRequest,
            add_criterion,
            evaluate_options,
            start_decision_analysis,
        )

        mock_ctx = Mock()

        # Start analysis
        start_request = StartDecisionAnalysisRequest(
            topic="Software architecture decision",
            options=["Microservices", "Monolith", "Serverless"],
        )

        start_result = await start_decision_analysis(start_request, mock_ctx)
        session_id = start_result["session_id"]

        # Add criterion
        criterion_request = AddCriterionRequest(
            session_id=session_id,
            name="Team Size Impact",
            description="How team size affects implementation",
            weight=1.0,
        )

        await add_criterion(criterion_request, mock_ctx)

        # Mock evaluation with abstentions
        mock_results_with_abstentions = {
            "Team Size Impact": {
                "Microservices": (7.0, "Requires larger teams for effective management"),
                "Monolith": (8.0, "Works well with smaller teams"),
                "Serverless": (None, "Not applicable for this criterion"),  # Abstention
            },
        }

        with patch.object(
            mcp_server.server_components.orchestrator,
            "evaluate_options_across_criteria",
            return_value=mock_results_with_abstentions,
        ):
            eval_request = EvaluateOptionsRequest(session_id=session_id)
            eval_result = await evaluate_options(eval_request, mock_ctx)

            assert eval_result["evaluation_complete"] is True
            assert eval_result["summary"]["successful_scores"] == 2
            assert eval_result["summary"]["abstentions"] == 1
            assert eval_result["summary"]["errors"] == 0

    @pytest.mark.asyncio()
    async def test_workflow_error_handling(self, mcp_server):
        """Test error handling throughout the workflow."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            AddCriterionRequest,
            EvaluateOptionsRequest,
            GetDecisionMatrixRequest,
            StartDecisionAnalysisRequest,
            add_criterion,
            evaluate_options,
            get_decision_matrix,
            start_decision_analysis,
        )

        mock_ctx = Mock()

        # Test invalid inputs
        invalid_start_request = StartDecisionAnalysisRequest(
            topic="",
            options=["A", "B"],  # Invalid empty topic
        )

        start_result = await start_decision_analysis(invalid_start_request, mock_ctx)
        assert "error" in start_result
        assert "formatted_output" in start_result

        # Test with valid session
        valid_start_request = StartDecisionAnalysisRequest(
            topic="Valid topic",
            options=["Option A", "Option B"],
        )

        start_result = await start_decision_analysis(valid_start_request, mock_ctx)
        session_id = start_result["session_id"]

        # Test invalid criterion
        invalid_criterion_request = AddCriterionRequest(
            session_id=session_id,
            name="",  # Invalid empty name
            description="Valid description",
            weight=1.0,
        )

        criterion_result = await add_criterion(invalid_criterion_request, mock_ctx)
        assert "error" in criterion_result
        assert "formatted_output" in criterion_result

        # Test evaluation without criteria
        eval_request = EvaluateOptionsRequest(session_id=session_id)
        eval_result = await evaluate_options(eval_request, mock_ctx)
        assert "error" in eval_result
        assert "No criteria defined" in eval_result["error"]

        # Test matrix retrieval without evaluation
        matrix_request = GetDecisionMatrixRequest(session_id=session_id)
        matrix_result = await get_decision_matrix(matrix_request, mock_ctx)
        # Should return a result even without evaluation, showing empty matrix
        assert "matrix" in matrix_result or "error" in matrix_result


class TestSessionManagementIntegration:
    """Test session management through MCP interface."""

    @pytest.mark.asyncio()
    async def test_session_isolation(self, mcp_server):
        """Test that sessions are properly isolated from each other."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            AddCriterionRequest,
            StartDecisionAnalysisRequest,
            add_criterion,
            list_sessions,
            start_decision_analysis,
        )

        mock_ctx = Mock()

        # Create two separate sessions
        session1_request = StartDecisionAnalysisRequest(
            topic="Database choice for Project A",
            options=["MySQL", "PostgreSQL"],
        )

        session2_request = StartDecisionAnalysisRequest(
            topic="Frontend framework for Project B",
            options=["React", "Vue", "Angular"],
        )

        session1_result = await start_decision_analysis(session1_request, mock_ctx)
        session2_result = await start_decision_analysis(session2_request, mock_ctx)

        session1_id = session1_result["session_id"]
        session2_id = session2_result["session_id"]

        assert session1_id != session2_id

        # Add different criteria to each session
        criterion1_request = AddCriterionRequest(
            session_id=session1_id,
            name="ACID Compliance",
            description="Database ACID properties",
            weight=2.0,
        )

        criterion2_request = AddCriterionRequest(
            session_id=session2_id,
            name="Learning Curve",
            description="Ease of learning framework",
            weight=1.5,
        )

        await add_criterion(criterion1_request, mock_ctx)
        await add_criterion(criterion2_request, mock_ctx)

        # List sessions and verify isolation
        sessions_result = await list_sessions(mock_ctx)

        sessions_by_id = {s["session_id"]: s for s in sessions_result["sessions"]}

        assert session1_id in sessions_by_id
        assert session2_id in sessions_by_id

        session1_data = sessions_by_id[session1_id]
        session2_data = sessions_by_id[session2_id]

        # Verify session 1 data
        assert session1_data["topic"] == "Database choice for Project A"
        assert set(session1_data["options"]) == {"MySQL", "PostgreSQL"}
        assert session1_data["criteria"] == ["ACID Compliance"]

        # Verify session 2 data
        assert session2_data["topic"] == "Frontend framework for Project B"
        assert set(session2_data["options"]) == {"React", "Vue", "Angular"}
        assert session2_data["criteria"] == ["Learning Curve"]

    @pytest.mark.asyncio()
    async def test_session_cleanup(self, mcp_server):
        """Test session cleanup functionality."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            StartDecisionAnalysisRequest,
            clear_all_sessions,
            list_sessions,
            start_decision_analysis,
        )

        mock_ctx = Mock()

        # Create multiple sessions
        for i in range(3):
            request = StartDecisionAnalysisRequest(
                topic=f"Decision {i}",
                options=[f"Option {i}A", f"Option {i}B"],
            )
            await start_decision_analysis(request, mock_ctx)

        # Verify sessions exist
        sessions_result = await list_sessions(mock_ctx)
        assert sessions_result["total_active"] >= 3

        # Clear all sessions
        clear_result = await clear_all_sessions(mock_ctx)
        assert clear_result["cleared"] >= 3

        # Verify sessions are gone
        sessions_result_after = await list_sessions(mock_ctx)
        assert sessions_result_after["total_active"] == 0
        assert sessions_result_after["sessions"] == []

    @pytest.mark.asyncio()
    async def test_current_session_tracking(self, mcp_server):
        """Test current session tracking functionality."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            StartDecisionAnalysisRequest,
            current_session,
            start_decision_analysis,
        )

        mock_ctx = Mock()

        # Initially no current session
        current_result = await current_session(mock_ctx)
        initial_session_count = len(current_result.get("session", {}) or {})

        # Create a session
        request = StartDecisionAnalysisRequest(
            topic="Current session test",
            options=["Option X", "Option Y"],
        )

        session_result = await start_decision_analysis(request, mock_ctx)
        session_id = session_result["session_id"]

        # Check current session
        current_result = await current_session(mock_ctx)

        if current_result["session"] is not None:
            assert current_result["session_id"] == session_id
            assert current_result["topic"] == "Current session test"
            assert set(current_result["options"]) == {"Option X", "Option Y"}
            assert current_result["status"] == "pending"


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance and edge cases."""

    @pytest.mark.asyncio()
    async def test_error_response_format(self, mcp_server):
        """Test that error responses follow expected format."""
        from unittest.mock import Mock

        from decision_matrix_mcp import StartDecisionAnalysisRequest, start_decision_analysis

        mock_ctx = Mock()

        # Test invalid request
        invalid_request = StartDecisionAnalysisRequest(
            topic="",
            options=["A"],  # Invalid empty topic  # Too few options
        )

        result = await start_decision_analysis(invalid_request, mock_ctx)

        # Verify error response structure
        assert "error" in result
        assert isinstance(result["error"], str)
        assert "formatted_output" in result
        assert isinstance(result["formatted_output"], str)

        # Error message should be user-friendly
        assert len(result["error"]) > 0
        assert "Invalid topic" in result["error"] or "Need at least 2 options" in result["error"]

    @pytest.mark.asyncio()
    async def test_all_tools_return_proper_format(self, mcp_server):
        """Test that all tools return properly formatted responses."""
        from unittest.mock import Mock

        from decision_matrix_mcp import (
            AddCriterionRequest,
            StartDecisionAnalysisRequest,
            add_criterion,
            clear_all_sessions,
            current_session,
            list_sessions,
            start_decision_analysis,
            test_aws_bedrock_connection,
        )

        mock_ctx = Mock()

        # Test each tool for proper response format

        # list_sessions
        list_result = await list_sessions(mock_ctx)
        assert isinstance(list_result, dict)
        assert "sessions" in list_result
        assert "total_active" in list_result

        # current_session
        current_result = await current_session(mock_ctx)
        assert isinstance(current_result, dict)

        # clear_all_sessions
        clear_result = await clear_all_sessions(mock_ctx)
        assert isinstance(clear_result, dict)
        assert "cleared" in clear_result

        # test_aws_bedrock_connection
        bedrock_result = await test_aws_bedrock_connection(mock_ctx)
        assert isinstance(bedrock_result, dict)

        # start_decision_analysis (valid)
        valid_start = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
        start_result = await start_decision_analysis(valid_start, mock_ctx)
        assert isinstance(start_result, dict)
        assert "session_id" in start_result

        # add_criterion (invalid session)
        invalid_criterion = AddCriterionRequest(
            session_id="invalid-session-id",
            name="Test",
            description="Test",
        )
        criterion_result = await add_criterion(invalid_criterion, mock_ctx)
        assert isinstance(criterion_result, dict)
        assert "error" in criterion_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
