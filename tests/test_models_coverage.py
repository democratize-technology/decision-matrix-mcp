"""Additional models tests for 100% coverage"""

from datetime import datetime, timezone

from decision_matrix_mcp.models import DecisionSession


class TestModelsFullCoverage:
    """Additional tests to achieve 100% coverage in models.py"""

    def test_get_decision_matrix_no_recommendation(self):
        """Test decision matrix when no scores are available"""
        session = DecisionSession(
            topic="Test Decision", session_id="test-123", created_at=datetime.now(timezone.utc)
        )

        # Add options and at least one criterion
        session.add_option("Option A")
        session.add_option("Option B")

        # Add a criterion but don't evaluate
        from decision_matrix_mcp.models import Criterion

        criterion = Criterion(name="Cost", weight=1.0, description="Cost analysis")
        session.add_criterion(criterion)

        # Get matrix without any evaluations
        matrix = session.get_decision_matrix()

        # Should have no clear recommendation when all scores are 0
        assert "recommendation" in matrix
        # Both options have 0 score, so it's a close race
        assert (
            "Close race" in matrix["recommendation"]
            or matrix["recommendation"] == "No clear recommendation available"
        )
