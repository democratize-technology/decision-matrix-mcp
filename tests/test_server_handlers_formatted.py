"""Tests for server handlers with formatted output"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

from mcp.server.fastmcp import Context
import pytest

from decision_matrix_mcp import (
    AddCriterionRequest,
    AddOptionRequest,
    EvaluateOptionsRequest,
    GetDecisionMatrixRequest,
    StartDecisionAnalysisRequest,
    add_criterion,
    add_option,
    evaluate_options,
    get_decision_matrix,
    list_sessions,
    start_decision_analysis,
)
from decision_matrix_mcp.exceptions import ValidationError

# Mock context for all tests
mock_ctx = Mock(spec=Context)


class TestFormattedResponses:
    """Test that all handlers return formatted_output field"""

    @pytest.mark.asyncio()
    async def test_start_decision_analysis_with_formatted_output(self):
        """Test start_decision_analysis includes formatted output"""
        request = StartDecisionAnalysisRequest(
            topic="Choose a database",
            options=["PostgreSQL", "MongoDB", "DynamoDB"],
            initial_criteria=[
                {"name": "Performance", "description": "Speed and efficiency", "weight": 2.0},
                {"name": "Cost", "description": "Total cost of ownership", "weight": 1.5},
            ],
        )

        result = await start_decision_analysis(request, mock_ctx)

        # Check basic response structure
        assert "session_id" in result
        assert result["topic"] == "Choose a database"
        assert len(result["options"]) == 3
        assert len(result["criteria_added"]) == 2

        # Check formatted output
        assert "formatted_output" in result
        formatted = result["formatted_output"]
        assert "# 🎯 Decision Analysis Session Created" in formatted
        assert "**Topic**: Choose a database" in formatted
        assert "1. **PostgreSQL**" in formatted
        assert "2. **MongoDB**" in formatted
        assert "3. **DynamoDB**" in formatted
        assert "## ⚖️ Initial Criteria" in formatted
        assert "- Performance" in formatted
        assert "- Cost" in formatted

    @pytest.mark.asyncio()
    async def test_add_criterion_with_formatted_output(self):
        """Test add_criterion includes formatted output"""
        # First create a session
        start_request = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
        start_result = await start_decision_analysis(start_request, mock_ctx)
        session_id = start_result["session_id"]

        # Add criterion
        request = AddCriterionRequest(
            session_id=session_id,
            name="Scalability",
            description="How well it scales",
            weight=2.5,
        )

        result = await add_criterion(request, mock_ctx)

        # Check formatted output
        assert "formatted_output" in result
        formatted = result["formatted_output"]
        assert "## ✅ Added Criterion: **Scalability**" in formatted
        assert "**Weight**: 2.5x importance" in formatted
        assert "💡 **Ready to evaluate**" in formatted

    @pytest.mark.asyncio()
    async def test_evaluate_options_with_formatted_output(self):
        """Test evaluate_options includes formatted output"""
        # Create session with criteria
        start_request = StartDecisionAnalysisRequest(
            topic="Test",
            options=["Option A", "Option B"],
            initial_criteria=[{"name": "Test", "description": "Test criterion", "weight": 1.0}],
        )
        start_result = await start_decision_analysis(start_request, mock_ctx)
        session_id = start_result["session_id"]

        # Mock orchestrator
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components

            # Mock successful evaluation
            mock_orchestrator = AsyncMock()
            mock_orchestrator.evaluate_options_across_criteria.return_value = {
                "Test": {
                    "Option A": (8.0, "Good option"),
                    "Option B": (6.0, "Moderate option"),
                },
            }
            mock_components.orchestrator = mock_orchestrator

            # Mock session manager to return our session
            mock_session_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.options = {"Option A": MagicMock(), "Option B": MagicMock()}
            mock_session.criteria = {"Test": MagicMock()}
            mock_session.threads = {"Test": MagicMock()}
            mock_session_manager.get_session.return_value = mock_session
            mock_components.session_manager = mock_session_manager

            # Mock formatter
            mock_formatter = MagicMock()
            mock_formatter.format_evaluation_complete.return_value = "# ✨ Evaluation Complete!"
            mock_formatter.format_error.return_value = "## ❌ Error"
            mock_components.formatter = mock_formatter

            request = EvaluateOptionsRequest(session_id=session_id)
            result = await evaluate_options(request, mock_ctx)

            # Check formatted output
            assert "formatted_output" in result
            assert result["formatted_output"] == "# ✨ Evaluation Complete!"

    @pytest.mark.asyncio()
    async def test_get_decision_matrix_with_formatted_output(self):
        """Test get_decision_matrix includes formatted output"""
        # Create and evaluate session
        start_request = StartDecisionAnalysisRequest(
            topic="Choose a language",
            options=["Python", "JavaScript"],
            initial_criteria=[{"name": "Ease", "description": "Ease of use", "weight": 2.0}],
        )
        start_result = await start_decision_analysis(start_request, mock_ctx)
        session_id = start_result["session_id"]

        # Mock the matrix generation
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components

            # Mock session with matrix data
            mock_session = MagicMock()
            mock_session.get_decision_matrix.return_value = {
                "topic": "Choose a language",
                "rankings": [
                    {"option": "Python", "weighted_total": 8.5, "breakdown": []},
                    {"option": "JavaScript", "weighted_total": 7.0, "breakdown": []},
                ],
                "recommendation": "Python is the winner",
                "criteria_weights": {"Ease": 2.0},
            }
            mock_session.created_at = MagicMock(isoformat=lambda: "2025-01-01T12:00:00")
            mock_session.evaluations = []
            mock_session.options = {}
            mock_session.criteria = {}

            mock_session_manager = MagicMock()
            mock_session_manager.get_session.return_value = mock_session
            mock_components.session_manager = mock_session_manager

            # Mock formatter
            mock_formatter = MagicMock()
            mock_formatter.format_decision_matrix.return_value = (
                "# 🎯 Decision Matrix: Choose a language\n### 🥇 **Winner: Python**"
            )
            mock_formatter.format_error.return_value = "## ❌ Error"
            mock_components.formatter = mock_formatter

            request = GetDecisionMatrixRequest(session_id=session_id)
            result = await get_decision_matrix(request, mock_ctx)

            # Check formatted output
            assert "formatted_output" in result
            assert "# 🎯 Decision Matrix: Choose a language" in result["formatted_output"]
            assert "### 🥇 **Winner: Python**" in result["formatted_output"]

    @pytest.mark.asyncio()
    async def test_list_sessions_with_formatted_output(self):
        """Test list_sessions includes formatted output"""
        result = await list_sessions(mock_ctx)

        # Check formatted output exists
        assert "formatted_output" in result
        formatted = result["formatted_output"]

        # Should show empty state or active sessions
        assert "Active Decision Sessions" in formatted or "No Active Sessions" in formatted

    @pytest.mark.asyncio()
    async def test_error_responses_with_formatted_output(self):
        """Test error responses include formatted output"""
        # Test validation error
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components

            mock_session_manager = MagicMock()
            mock_session_manager.create_session.side_effect = ValidationError(
                "Invalid",
                "Bad input",
            )
            mock_components.session_manager = mock_session_manager

            mock_formatter = MagicMock()
            mock_formatter.format_error.return_value = (
                "## ❌ Error Encountered\n**Issue**: Bad input"
            )
            mock_components.formatter = mock_formatter

            request = StartDecisionAnalysisRequest(topic="Test", options=["A"])
            result = await start_decision_analysis(request, mock_ctx)

            assert "error" in result
            assert "formatted_output" in result
            assert "## ❌ Error Encountered" in result["formatted_output"]

    @pytest.mark.asyncio()
    async def test_add_option_with_formatted_output(self):
        """Test add_option includes formatted output"""
        # Create session first
        start_request = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
        start_result = await start_decision_analysis(start_request, mock_ctx)
        session_id = start_result["session_id"]

        # Add new option
        request = AddOptionRequest(
            session_id=session_id,
            option_name="Option C",
            description="Third option",
        )

        result = await add_option(request, mock_ctx)

        # Check formatted output
        assert "formatted_output" in result
        formatted = result["formatted_output"]
        assert "## ✅ Added Option: **Option C**" in formatted
        assert "*Third option*" in formatted
        assert "⚡ **Action Required**" in formatted

    @pytest.mark.asyncio()
    async def test_session_not_found_error_formatting(self):
        """Test session not found error has helpful formatting"""
        request = AddCriterionRequest(
            session_id="invalid-session-id",
            name="Test",
            description="Test",
        )

        result = await add_criterion(request, mock_ctx)

        assert "error" in result
        assert "formatted_output" in result
        formatted = result["formatted_output"]
        assert "## ❌ Error Encountered" in formatted
        assert "💡 **Suggestions**:" in formatted
        assert "- Check the session ID is correct" in formatted
        assert "- Session may have expired (30 min timeout)" in formatted

    @pytest.mark.asyncio()
    async def test_no_options_error_formatting(self):
        """Test no options error has helpful formatting"""
        # Create session without options
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components

            # Mock session with no options
            mock_session = MagicMock()
            mock_session.options = {}
            mock_session.criteria = {"Test": MagicMock()}

            mock_session_manager = MagicMock()
            mock_session_manager.get_session.return_value = mock_session
            mock_components.session_manager = mock_session_manager

            mock_formatter = MagicMock()
            mock_formatter.format_error.return_value = (
                "## ❌ Error Encountered\n"
                "**Issue**: No options to evaluate. Add options first.\n"
                "💡 **Next step**: Add options to evaluate first"
            )
            mock_components.formatter = mock_formatter

            request = EvaluateOptionsRequest(session_id="test")
            result = await evaluate_options(request, mock_ctx)

            assert "error" in result
            assert "formatted_output" in result
            assert "💡 **Next step**: Add options to evaluate first" in result["formatted_output"]

    @pytest.mark.asyncio()
    async def test_no_criteria_error_formatting(self):
        """Test no criteria error has helpful formatting"""
        # Create session without criteria
        start_request = StartDecisionAnalysisRequest(topic="Test", options=["A", "B"])
        start_result = await start_decision_analysis(start_request, mock_ctx)
        session_id = start_result["session_id"]

        request = EvaluateOptionsRequest(session_id=session_id)
        result = await evaluate_options(request, mock_ctx)

        assert "error" in result
        assert "formatted_output" in result
        assert "💡 **Next step**: Add evaluation criteria first" in result["formatted_output"]
