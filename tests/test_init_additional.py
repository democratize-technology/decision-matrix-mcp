"""Additional tests for __init__.py to achieve 100% coverage."""

from unittest.mock import Mock, patch

import pytest

from decision_matrix_mcp import add_criterion, create_server_components

# Create server components for testing
test_components = create_server_components()


@pytest.fixture(autouse=True)
def patch_server_components(monkeypatch):
    """Automatically patch get_server_components to return our test components"""
    monkeypatch.setattr("decision_matrix_mcp.get_server_components", lambda: test_components)


class TestInitAdditionalCoverage:
    """Additional test cases for __init__.py coverage"""

    @pytest.mark.asyncio()
    async def test_add_criterion_validation_error_path(self):
        """Test lines 218-219: ValidationError in add_criterion"""
        from mcp.server.fastmcp import Context

        from decision_matrix_mcp import AddCriterionRequest

        # Get session manager from server components
        components = test_components
        session_manager = components.session_manager

        # Create a valid session first
        session = session_manager.create_session("Test Topic", ["A", "B"])
        session_id = session.session_id

        # Create request with invalid weight
        request = AddCriterionRequest(
            session_id=session_id,
            name="TestCriterion",
            description="Test description",
            weight=15.0,  # Invalid - too high
        )

        # Create a mock context
        ctx = Mock(spec=Context)

        # Call add_criterion which should trigger validation error
        result = await add_criterion(request, ctx)

        assert "error" in result
        assert "Invalid weight" in result["error"]
        assert "0.1" in result["error"] and "10" in result["error"]

        # Cleanup
        session_manager.remove_session(session_id)

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
