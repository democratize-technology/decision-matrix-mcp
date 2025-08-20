#!/usr/bin/env python3
"""
Bug fix validation test that demonstrates the exact scenario that was broken
and proves it now works correctly with the cached property fix.

This test simulates the exact workflow that was failing before the fix and
validates that it now works correctly.
"""

from datetime import datetime, timezone
from uuid import uuid4

from decision_matrix_mcp.models import Criterion, DecisionSession, ModelBackend


class TestBugFixValidation:
    """Validate the specific bug fix works in the original failing scenario."""

    def test_original_bug_scenario_now_works(self):
        """
        This test reproduces the exact scenario that was failing before the fix.

        Before the fix: Session creation with criteria addition would fail when
        _invalidate_cache() was called with hasattr(self, "decision_matrix") because
        hasattr() would trigger the @cached_property getter prematurely.

        After the fix: Using "decision_matrix" in self.__dict__ avoids triggering
        the property and allows proper cache invalidation.
        """
        # STEP 1: Create a new decision session (this was always working)
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Framework Selection",
        )

        # STEP 2: Add some options (this was always working)
        session.add_option("React", "Facebook's React framework")
        session.add_option("Vue", "Progressive Vue.js framework")
        session.add_option("Angular", "Google's Angular framework")

        # STEP 3: Add criteria (THIS IS WHERE THE BUG WOULD OCCUR)
        # Each add_criterion() call triggers _invalidate_cache() which was broken

        # First criterion addition - would fail here before the fix
        session.add_criterion(
            Criterion(
                name="Performance",
                description="Runtime performance and speed",
                weight=2.5,
                model_backend=ModelBackend.BEDROCK,
            ),
        )

        # Verify first addition worked
        assert len(session.criteria) == 1
        assert "Performance" in session.criteria
        assert "decision_matrix" not in session.__dict__  # Property not triggered

        # Second criterion addition - would also fail before the fix
        session.add_criterion(
            Criterion(
                name="Developer Experience",
                description="Ease of development and tooling",
                weight=2.0,
                model_backend=ModelBackend.LITELLM,
                model_name="gpt-4",
            ),
        )

        # Verify second addition worked
        assert len(session.criteria) == 2
        assert "Developer Experience" in session.criteria
        assert "decision_matrix" not in session.__dict__  # Still not triggered

        # Third criterion addition - would fail before the fix
        session.add_criterion(
            Criterion(
                name="Community Support",
                description="Ecosystem and community resources",
                weight=1.5,
                model_backend=ModelBackend.OLLAMA,
            ),
        )

        # Verify third addition worked
        assert len(session.criteria) == 3
        assert "Community Support" in session.criteria
        assert "decision_matrix" not in session.__dict__  # Still not triggered

        # STEP 4: Now access the decision matrix property (should work correctly)
        matrix = session.decision_matrix

        # Verify the property was cached and works correctly
        assert "decision_matrix" in session.__dict__  # Now it's cached

        # Without evaluations, the matrix will be in "not ready" state
        if "status" in matrix:
            # Expected case - matrix not ready until evaluation
            assert matrix["status"] == "setup"
            assert "message" in matrix
            assert matrix["has_criteria"] is True
            assert matrix["has_evaluations"] is False
        else:
            # If somehow ready, check normal structure
            assert "matrix" in matrix
            assert "rankings" in matrix
            assert "criteria_weights" in matrix

            # Verify all our data is present
            assert len(matrix["matrix"]) == 3  # Three options
            assert len(matrix["criteria_weights"]) == 3  # Three criteria
            assert matrix["criteria_weights"]["Performance"] == 2.5
            assert matrix["criteria_weights"]["Developer Experience"] == 2.0
            assert matrix["criteria_weights"]["Community Support"] == 1.5

        # STEP 5: Add another criterion to trigger cache invalidation again
        session.add_criterion(
            Criterion(
                name="Learning Curve",
                description="Ease of learning for new developers",
                weight=1.0,
            ),
        )

        # Cache should be invalidated (this would also fail before the fix)
        assert "decision_matrix" not in session.__dict__

        # Access property again to verify it recomputes correctly
        new_matrix = session.decision_matrix
        if "status" in new_matrix:
            # Still in not-ready state (expected without evaluations)
            assert new_matrix["has_criteria"] is True
            assert len(session.criteria) == 4  # Four criteria now
            assert "Learning Curve" in session.criteria
        else:
            # If somehow ready, check the data
            assert len(new_matrix["criteria_weights"]) == 4  # Four criteria now
            assert "Learning Curve" in new_matrix["criteria_weights"]

    def test_multiple_rapid_criteria_additions_work(self):
        """
        Test that rapid multiple criteria additions work (stress test the fix).

        This simulates a scenario where many criteria are added quickly,
        causing multiple rapid cache invalidations.
        """
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Rapid Criteria Test",
        )

        session.add_option("Option A")
        session.add_option("Option B")

        # Add 10 criteria rapidly
        for i in range(10):
            session.add_criterion(
                Criterion(
                    name=f"Criterion_{i}",
                    description=f"Test criterion number {i}",
                    weight=float(i + 1),
                    model_backend=ModelBackend.BEDROCK,
                ),
            )

            # Each addition should work without error
            assert len(session.criteria) == i + 1
            assert "decision_matrix" not in session.__dict__

        # Final verification
        assert len(session.criteria) == 10

        # Property access should work
        matrix = session.decision_matrix
        if "status" in matrix:
            # Without evaluations, will be in setup state
            assert matrix["has_criteria"] is True
            assert len(session.criteria) == 10
        else:
            assert len(matrix["criteria_weights"]) == 10

    def test_cache_invalidation_timing_is_correct(self):
        """
        Test that the timing of cache invalidation is correct with the fix.

        This verifies that:
        1. Property is not triggered during cache invalidation
        2. Property works correctly when explicitly accessed
        3. Cache invalidation properly removes cached values
        """
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Cache Timing Test",
        )

        session.add_option("Test Option")

        # Add initial criterion
        session.add_criterion(
            Criterion(name="Initial", description="Initial criterion", weight=1.0),
        )

        # Verify not cached yet
        assert "decision_matrix" not in session.__dict__

        # Access property to cache it
        matrix1 = session.decision_matrix
        assert "decision_matrix" in session.__dict__

        # Add another criterion (triggers cache invalidation)
        session.add_criterion(Criterion(name="Second", description="Second criterion", weight=2.0))

        # Should be invalidated immediately
        assert "decision_matrix" not in session.__dict__

        # Access again
        matrix2 = session.decision_matrix
        assert "decision_matrix" in session.__dict__

        # Should be different objects (recomputed)
        assert matrix1 is not matrix2
        if "status" in matrix2:
            # Still in setup state without evaluations
            assert matrix2["has_criteria"] is True
            assert len(session.criteria) == 2
        else:
            assert len(matrix2["criteria_weights"]) == 2

    def test_session_manager_integration_workflow(self):
        """
        Test the integration with session manager workflows that were failing.

        This simulates the exact workflow used by the MCP handlers that was
        failing before the fix.
        """
        # Simulate session_manager.create_session with initial criteria
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Integration Test",
        )

        # Add options (from start_decision_analysis request)
        for option in ["AWS", "GCP", "Azure"]:
            session.add_option(option, f"{option} cloud platform")

        # Simulate process_initial_criteria workflow
        initial_criteria = [
            {
                "name": "Cost",
                "description": "Monthly cost analysis",
                "weight": 2.0,
                "model_backend": "bedrock",
            },
            {
                "name": "Performance",
                "description": "Latency and throughput",
                "weight": 2.5,
                "model_backend": "litellm",
                "model_name": "gpt-4",
            },
            {
                "name": "Reliability",
                "description": "Uptime and SLA guarantees",
                "weight": 3.0,
                "model_backend": "ollama",
            },
        ]

        # Process each criterion (this was failing before the fix)
        for spec in initial_criteria:
            criterion = Criterion(
                name=spec["name"],
                description=spec["description"],
                weight=spec["weight"],
                model_backend=ModelBackend(spec["model_backend"]),
                model_name=spec.get("model_name"),
            )

            # This add_criterion call was failing before the fix
            session.add_criterion(criterion)

        # Verify complete workflow succeeded
        assert len(session.options) == 3
        assert len(session.criteria) == 3
        assert len(session.threads) == 3  # Threads created for each criterion

        # Decision matrix should work
        matrix = session.decision_matrix
        if "status" in matrix:
            # Without evaluations, will be in setup state
            assert matrix["has_criteria"] is True
            assert len(session.options) == 3  # All options
            assert len(session.criteria) == 3  # All criteria
        else:
            assert len(matrix["matrix"]) == 3  # All options
            assert len(matrix["criteria_weights"]) == 3  # All criteria


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
