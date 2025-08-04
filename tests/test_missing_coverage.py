"""Tests for 100% coverage of remaining modules"""

import pytest
from unittest.mock import MagicMock, Mock, patch

from decision_matrix_mcp import __main__, SessionError, clear_all_sessions
from mcp.server.fastmcp import Context
from decision_matrix_mcp.models import Option, DecisionSession, Score
from decision_matrix_mcp.session_manager import SessionManager


# Mock context for all tests
mock_ctx = Mock(spec=Context)

class TestMainModule:
    """Test __main__.py module"""

    def test_main_module_execution(self):
        """Test that main function can be imported"""
        # Test that the main function can be imported without error
        from decision_matrix_mcp import main
        assert callable(main)


class TestModelEdgeCases:
    """Test edge cases in models.py"""

    def test_option_get_score_breakdown_with_missing_criteria(self):
        """Test score breakdown when criteria dict doesn't have all criteria"""
        option = Option(name="Test")
        
        # Add score for a criterion
        score = Score(
            criterion_name="Missing",
            option_name="Test",
            score=8.0,
            justification="Good"
        )
        option.add_score(score)
        
        # Get breakdown with empty criteria dict (criterion not in dict)
        breakdown = option.get_score_breakdown({})
        
        # Should return empty list since criterion not in criteria dict
        assert breakdown == []

    def test_decision_session_get_matrix_with_none_scores(self):
        """Test matrix generation with None scores in edge cases"""
        session = DecisionSession(
            session_id="test",
            created_at=MagicMock(),
            topic="Test"
        )
        
        # Add option and criterion
        session.add_option("Option A")
        from decision_matrix_mcp.models import Criterion
        criterion = Criterion("Test", "Test criterion")
        session.add_criterion(criterion)
        
        # Add score with None value but also None for score attribute check
        score = Score(
            criterion_name="Test",
            option_name="Option A", 
            score=None,
            justification="Abstained"
        )
        # Manually set to test the None check branch
        score.score = 0.0  # This will test the "score is not None" but value is falsy
        session.options["Option A"].add_score(score)
        
        matrix = session.get_decision_matrix()
        
        # Should handle the falsy but not None score
        assert matrix["matrix"]["Option A"]["Test"]["raw_score"] == 0.0
        assert matrix["matrix"]["Option A"]["Test"]["weighted_score"] == 0.0


class TestSessionManagerEdgeCases:
    """Test edge cases in session_manager.py"""

    def test_session_manager_stats_calculation(self):
        """Test get_stats when sessions have been removed"""
        manager = SessionManager(max_sessions=5)
        
        # Create and remove some sessions
        session1 = manager.create_session("Test 1")
        session2 = manager.create_session("Test 2")
        manager.remove_session(session1.session_id)
        
        stats = manager.get_stats()
        
        assert stats["sessions_created"] == 2
        assert stats["active_sessions"] == 1
        assert "sessions_cleaned" in stats
        assert "max_concurrent" in stats


class TestClearAllSessions:
    """Test clear_all_sessions function"""

    @pytest.mark.asyncio
    async def test_clear_all_sessions_success(self):
        """Test successful clearing of all sessions"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components
            
            # Mock session manager with active sessions
            mock_session_manager = MagicMock()
            mock_session_manager.list_active_sessions.return_value = {
                "session1": MagicMock(),
                "session2": MagicMock(),
                "session3": MagicMock(),
            }
            mock_session_manager.remove_session.side_effect = [True, True, False]  # One fails
            mock_session_manager.get_stats.return_value = {"active_sessions": 1}
            mock_components.session_manager = mock_session_manager
            
            result = await clear_all_sessions(mock_ctx)
            
            assert result["cleared"] == 2  # Only 2 succeeded
            assert "Cleared 2 active sessions" in result["message"]
            assert result["stats"]["active_sessions"] == 1

    @pytest.mark.asyncio
    async def test_clear_all_sessions_error(self):
        """Test error handling in clear_all_sessions"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components
            
            # Mock session manager that raises exception
            mock_session_manager = MagicMock()
            mock_session_manager.list_active_sessions.side_effect = Exception("Database error")
            mock_components.session_manager = mock_session_manager
            
            mock_formatter = MagicMock()
            mock_formatter.format_error.return_value = "## ❌ Error"
            mock_components.formatter = mock_formatter
            
            result = await clear_all_sessions(mock_ctx)
            
            assert "error" in result
            assert "Failed to clear sessions" in result["error"]


class TestSessionErrorHandling:
    """Test SessionError handling in handlers"""

    @pytest.mark.asyncio  
    async def test_add_criterion_session_error(self):
        """Test SessionError handling in add_criterion"""
        from decision_matrix_mcp import add_criterion, AddCriterionRequest
        
        # Create a valid session first
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            mock_components = MagicMock()
            mock_get_components.return_value = mock_components
            
            # Mock session manager returns session
            mock_session = MagicMock()
            mock_session.criteria = {}
            mock_session_manager = MagicMock()
            mock_session_manager.get_session.return_value = mock_session
            mock_components.session_manager = mock_session_manager
            
            # Make add_criterion raise SessionError
            mock_session.add_criterion.side_effect = SessionError(
                "Session error", "Session is locked"
            )
            
            request = AddCriterionRequest(
                session_id="test",
                name="Test", 
                description="Test",
                weight=1.0
            )
            
            result = await add_criterion(request, mock_ctx)
            
            assert "error" in result
        assert result["error"] == "Session is locked"
        assert "formatted_output" in result


class TestValidationDecorator:
    """Test validation decorator edge cases"""

    def test_validate_criteria_spec_with_invalid_weight_type(self):
        """Test criteria spec validation with non-numeric weight"""
        from decision_matrix_mcp.validation_decorators import validate_criteria_spec
        
        # Test with string weight that can't be converted
        criteria = [{"name": "Test", "description": "Test", "weight": "invalid"}]
        result = validate_criteria_spec(criteria)
        
        assert result is not None
        assert "error" in result
        assert "Invalid weight" in result["error"]


class TestFormatterMissingLines:
    """Test missing lines in formatter"""

    def test_formatter_score_bar_zero_max_score(self):
        """Test score bar with zero max score"""
        from decision_matrix_mcp.formatting import DecisionFormatter
        
        formatter = DecisionFormatter()
        bar = formatter._create_score_bar(5.0, 0.0)  # max_score = 0
        assert bar == ""  # Should return empty string

    def test_formatter_invalid_verbosity_level(self):
        """Test setting invalid verbosity level"""
        from decision_matrix_mcp.formatting import DecisionFormatter
        
        formatter = DecisionFormatter()
        with pytest.raises(ValueError) as exc_info:
            formatter.set_verbosity("invalid_level")
        assert "Invalid verbosity level: invalid_level" in str(exc_info.value)

    def test_formatter_many_errors_truncation(self):
        """Test error list truncation in evaluation complete"""
        from decision_matrix_mcp.formatting import DecisionFormatter
        
        formatter = DecisionFormatter()
        eval_data = {
            "summary": {
                "options_evaluated": 3,
                "criteria_used": 2,
                "total_evaluations": 6,
                "successful_scores": 0,
                "abstentions": 0,
                "errors": 10,
            },
            "errors": [f"Error {i}" for i in range(10)]  # 10 errors
        }
        
        output = formatter.format_evaluation_complete(eval_data)
        
        # Should show first 5 and "...and 5 more"
        assert "Error 0" in output
        assert "Error 4" in output
        assert "Error 5" not in output  # Should be truncated
        assert "...and 5 more" in output

    def test_formatter_strong_winner_insight(self):
        """Test strong winner insight in decision matrix"""
        from decision_matrix_mcp.formatting import DecisionFormatter
        
        formatter = DecisionFormatter()
        matrix_data = {
            "topic": "Test",
            "rankings": [{
                "option": "Winner",
                "weighted_total": 9.5,  # > 8.0 threshold
                "breakdown": []
            }],
            "recommendation": "Winner wins",
            "criteria_weights": {},
            "session_id": "test"
        }
        
        output = formatter.format_decision_matrix(matrix_data)
        assert "- 🌟 **Strong winner**" in output

    def test_formatter_detailed_justifications(self):
        """Test detailed mode shows justifications"""
        from decision_matrix_mcp.formatting import DecisionFormatter
        
        formatter = DecisionFormatter(verbosity=DecisionFormatter.DETAILED)
        matrix_data = {
            "topic": "Test",
            "rankings": [{
                "option": "Test",
                "weighted_total": 8.0,
                "breakdown": [{
                    "criterion": "Cost",
                    "weight": 2.0,
                    "raw_score": 8.0,
                    "weighted_score": 16.0,
                    "justification": "Very cost effective solution with excellent ROI and low maintenance costs over time",
                    "abstained": False
                }]
            }],
            "recommendation": "Test wins",
            "criteria_weights": {"Cost": 2.0},
            "session_id": "test"
        }
        
        output = formatter.format_decision_matrix(matrix_data)
        # Should show truncated justification
        assert "Very cost effective solution with excellent ROI and low maintenance costs over t..." in output