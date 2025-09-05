"""Additional tests for __init__.py to achieve 100% coverage."""

from unittest.mock import patch

import pytest

from decision_matrix_mcp import add_criterion


class TestInitAdditionalCoverage:
    """Additional test cases for __init__.py coverage"""

    @pytest.mark.asyncio()
    async def test_add_criterion_validation_error_path(
        self, server_components, mock_context, monkeypatch
    ):
        """Test that add_criterion correctly validates weight ranges at MCP level"""
        # Use isolated server components from fixture
        session_manager = server_components.session_manager

        # Create a valid session first
        session = session_manager.create_session("Test Topic", ["A", "B"])
        session_id = session.session_id

        # Patch get_server_components to return our isolated components
        monkeypatch.setattr("decision_matrix_mcp.get_server_components", lambda: server_components)

        # Call add_criterion with invalid high weight (should be rejected)
        result = await add_criterion(
            session_id=session_id,
            name="TestCriterion",
            description="Test description",
            weight=15.0,  # High weight - exceeds max of 10.0
            ctx=mock_context,
        )

        # MCP functions DO validate weight ranges (updated implementation)
        assert "error" in result
        assert "Invalid weight: must be between 0.1 and 10.0" in result["error"]
        assert "Invalid weight" in result["context"]

        # Cleanup is handled by the server_components fixture

    def test_main_module_execution_coverage(self):
        """Test lines 490-491: Module execution as __main__"""
        # Instead of trying to execute as __main__, let's test the main() function directly
        from decision_matrix_mcp import main

        # Mock the mcp.run() to prevent actual server startup
        with patch("decision_matrix_mcp.mcp.run") as mock_run:
            # Call main function
            main()

            # Verify mcp.run was called
            mock_run.assert_called_once()
