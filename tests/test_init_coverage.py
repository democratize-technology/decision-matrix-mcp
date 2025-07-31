"""Additional __init__.py tests for 100% coverage"""

from unittest.mock import patch

import pytest

from decision_matrix_mcp import (
    AddCriterionRequest,
    AddOptionRequest,
    EvaluateOptionsRequest,
    GetDecisionMatrixRequest,
    add_criterion,
    add_option,
    evaluate_options,
    get_decision_matrix,
)
from decision_matrix_mcp.session_manager import SessionManager


class TestInitFullCoverage:
    """Additional tests to achieve 100% coverage in __init__.py"""

    @pytest.mark.asyncio
    async def test_add_criterion_validation_error(self):
        """Test add_criterion with validation error"""
        # Create a session
        session_manager = SessionManager()
        session = session_manager.create_session("Test topic")

        # Patch session_manager to return our session
        with patch("decision_matrix_mcp.session_manager", session_manager):
            # Test with validation error - invalid weight
            request = AddCriterionRequest(
                session_id=session.session_id,
                name="Test",
                weight=0.05,  # Below minimum
                description="Test",
            )

            result = await add_criterion(request)
            assert "error" in result
            assert "Invalid weight" in result["error"]

    @pytest.mark.asyncio
    async def test_add_criterion_unexpected_error(self):
        """Test add_criterion with unexpected error"""
        # Create a session
        session_manager = SessionManager()
        session = session_manager.create_session("Test topic")

        # Patch session_manager to raise unexpected error
        with patch("decision_matrix_mcp.session_manager", session_manager):
            with patch.object(session, "add_criterion", side_effect=Exception("Unexpected")):
                request = AddCriterionRequest(
                    session_id=session.session_id, name="Test", weight=1.0, description="Test"
                )

                result = await add_criterion(request)
                assert "error" in result
                assert "unexpected error" in result["error"]

    @pytest.mark.asyncio
    async def test_evaluate_options_session_not_found(self):
        """Test evaluate_options with session not found"""
        session_manager = SessionManager()

        with patch("decision_matrix_mcp.session_manager", session_manager):
            request = EvaluateOptionsRequest(session_id="nonexistent")
            result = await evaluate_options(request)

            assert "error" in result
            assert "not found or expired" in result["error"]

    @pytest.mark.asyncio
    async def test_get_decision_matrix_session_not_found(self):
        """Test get_decision_matrix with session not found"""
        session_manager = SessionManager()

        with patch("decision_matrix_mcp.session_manager", session_manager):
            request = GetDecisionMatrixRequest(session_id="nonexistent")
            result = await get_decision_matrix(request)

            assert "error" in result
            assert "not found or expired" in result["error"]

    @pytest.mark.asyncio
    async def test_add_option_session_not_found(self):
        """Test add_option with session not found"""
        session_manager = SessionManager()

        with patch("decision_matrix_mcp.session_manager", session_manager):
            request = AddOptionRequest(
                session_id="nonexistent", option_name="Option A", description="Test option"
            )
            result = await add_option(request)

            assert "error" in result
            assert "not found or expired" in result["error"]

    def test_main_as_module(self):
        """Test __name__ == '__main__' branch"""
        # The __main__ branch is covered by running the module
        # This test just verifies the code structure exists
        import decision_matrix_mcp

        # Read the source to verify the __main__ check exists
        with open(decision_matrix_mcp.__file__) as f:
            source = f.read()

        # Verify the __main__ check is present
        assert 'if __name__ == "__main__"' in source
        assert "main()" in source
