"""Tests for remaining coverage gaps"""

from unittest.mock import MagicMock, patch

import pytest

from decision_matrix_mcp.models import Criterion, CriterionThread
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestRemainingCoverage:
    """Tests to cover remaining gaps"""

    def test_orchestrator_parse_error_exception(self):
        """Test parse evaluation response error handling"""
        orchestrator = DecisionOrchestrator()

        # Create a response that will cause JSON decode error
        invalid_json = "{'invalid': json}"

        score, justification = orchestrator._parse_evaluation_response(invalid_json)

        # Should handle the exception and return error
        assert score is None
        assert "Could not parse score" in justification

    def test_init_validation_error_logging(self):
        """Test validation error message formatting"""
        from decision_matrix_mcp.exceptions import ValidationError

        # Create a validation error with a message
        error = ValidationError("Test validation failed", "Test validation failed")

        # The user_message property should return the message
        assert error.user_message == "Test validation failed"

    @pytest.mark.asyncio
    async def test_orchestrator_litellm_no_content(self):
        """Test litellm response with no content"""
        orchestrator = DecisionOrchestrator()

        criterion = Criterion(name="Cost", weight=1.0, description="Test", model_backend="litellm")
        thread = CriterionThread(id="test-1", criterion=criterion)

        # Mock response with None content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("litellm.acompletion", return_value=mock_response):
            response = await orchestrator._call_litellm(thread)

        assert response is None
