#!/usr/bin/env python3
"""
Focused tests for the Decision Matrix MCP session creation cached property bug fix.

This test suite validates the critical fix applied to models.py:684 where
`hasattr(self, "decision_matrix")` was changed to `"decision_matrix" in self.__dict__`
to prevent premature @cached_property triggering during session creation.

Critical Bug Fixed:
- Session creation and criteria addition would fail due to premature property evaluation
- The hasattr() call would trigger the @cached_property getter during cache invalidation
- Changed to __dict__ check to avoid property evaluation during cleanup

This focuses on the core bug and regression prevention.
"""

import contextlib
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from decision_matrix_mcp.models import Criterion, DecisionSession, ModelBackend


class TestCachedPropertyBugFixCore:
    """Core tests for the cached property bug fix."""

    def test_session_creation_and_criteria_addition_regression(self):
        """
        REGRESSION TEST: Ensure session creation + criteria addition doesn't fail.

        This was the original bug scenario: adding criteria after session creation
        would trigger cache invalidation, which used hasattr() to check for cached
        property, which would prematurely trigger @cached_property evaluation.
        """
        # Create session first
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Choose Framework",
        )

        # Verify initial state
        assert "decision_matrix" not in session.__dict__

        # Adding criteria should trigger cache invalidation but not break
        # This is where the original bug would occur
        session.add_criterion(
            Criterion(
                name="Performance",
                description="Runtime performance",
                weight=2.0,
                model_backend=ModelBackend.BEDROCK,
            ),
        )

        # Should succeed without exceptions
        assert len(session.criteria) == 1
        assert "Performance" in session.criteria
        assert "decision_matrix" not in session.__dict__  # Still not triggered

        # Add another criterion (more cache invalidations)
        session.add_criterion(
            Criterion(
                name="Cost",
                description="Development cost",
                weight=1.5,
                model_backend=ModelBackend.LITELLM,
            ),
        )

        # Should still work
        assert len(session.criteria) == 2
        assert "Cost" in session.criteria
        assert "decision_matrix" not in session.__dict__

    def test_cached_property_still_works_after_fix(self):
        """Test that @cached_property functionality is preserved after the fix."""
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Property Test",
        )

        # Add minimal setup for property to work
        session.add_option("Option A")
        session.add_criterion(Criterion(name="Test", description="Test criterion", weight=1.0))

        # First access should compute and cache the property
        assert "decision_matrix" not in session.__dict__
        matrix1 = session.decision_matrix
        assert "decision_matrix" in session.__dict__

        # Second access should return cached value (same object)
        matrix2 = session.decision_matrix
        assert matrix1 is matrix2

    def test_cache_invalidation_behavior_after_fix(self):
        """Test that cache invalidation still works correctly after the fix."""
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Cache Test",
        )

        # Setup data
        session.add_option("Option A")
        session.add_criterion(Criterion(name="Test", description="Test criterion", weight=1.0))

        # Cache the property
        matrix1 = session.decision_matrix
        assert "decision_matrix" in session.__dict__

        # Add another criterion (should invalidate cache via _invalidate_cache())
        # This is where the bug would occur - in the _invalidate_cache() method
        session.add_criterion(Criterion(name="Test2", description="Another criterion", weight=2.0))

        # Cache should be invalidated (this tests the fix at line 684)
        assert "decision_matrix" not in session.__dict__

        # Next access should recompute with new criterion
        matrix2 = session.decision_matrix
        assert matrix1 is not matrix2  # Different object (recomputed)

        # Check criteria depending on status
        if "criteria_weights" in matrix2:
            assert len(matrix2["criteria_weights"]) == 2
        else:
            # Still in setup state
            assert len(session.criteria) == 2

    def test_dict_check_vs_hasattr_fix_validation(self):
        """
        Test that validates the specific fix: __dict__ check vs hasattr().

        The fix changed the cache invalidation from:
        if hasattr(self, "decision_matrix"):
        to:
        if "decision_matrix" in self.__dict__:

        This test ensures the __dict__ approach works correctly.
        """
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Fix Validation Test",
        )

        # Setup minimal data
        session.add_option("Option A")
        session.add_criterion(Criterion(name="Test", description="Test criterion", weight=1.0))

        # Initially, property should not be in __dict__
        assert "decision_matrix" not in session.__dict__

        # The __dict__ check should return False and not trigger property
        has_property = "decision_matrix" in session.__dict__
        assert has_property is False
        assert "decision_matrix" not in session.__dict__  # Still not triggered

        # Access property explicitly to cache it
        _ = session.decision_matrix
        assert "decision_matrix" in session.__dict__

        # Now __dict__ check returns True
        has_property = "decision_matrix" in session.__dict__
        assert has_property is True

        # Manual invalidation using the fixed approach
        if "decision_matrix" in session.__dict__:
            with contextlib.suppress(AttributeError):
                delattr(session, "decision_matrix")

        # Should be removed from __dict__
        assert "decision_matrix" not in session.__dict__

    def test_multiple_cache_invalidations_dont_break(self):
        """Test that multiple rapid cache invalidations don't break the session."""
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Multiple Invalidations Test",
        )

        # Setup initial data
        session.add_option("Option A")
        session.add_criterion(Criterion(name="Initial", description="Initial", weight=1.0))

        # Cache the property
        _matrix1 = session.decision_matrix  # This should cache the property
        assert "decision_matrix" in session.__dict__

        # Add multiple criteria rapidly (each triggers cache invalidation)
        for i in range(5):
            session.add_criterion(
                Criterion(
                    name=f"Criterion_{i}",
                    description=f"Description {i}",
                    weight=float(i + 1),
                ),
            )
            # Each add_criterion calls _invalidate_cache() which uses the fixed code
            assert "decision_matrix" not in session.__dict__

        # Final property access should work
        final_matrix = session.decision_matrix
        if "criteria_weights" in final_matrix:
            assert len(final_matrix["criteria_weights"]) == 6  # Initial + 5 added
        else:
            # Still in setup state
            assert len(session.criteria) == 6

    def test_edge_case_delattr_exception_handling(self):
        """Test that the exception handling in _invalidate_cache is robust."""
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Exception Test",
        )

        # Manually set a property in __dict__
        session.__dict__["decision_matrix"] = {"fake": "matrix"}
        assert "decision_matrix" in session.__dict__

        # Test the try/except block around delattr in _invalidate_cache
        # Even if delattr fails, it shouldn't break the session
        with patch("builtins.delattr", side_effect=AttributeError("Mock error")):
            # This should not raise exception due to try/except block
            try:
                session._invalidate_cache()
                # The method should complete without exceptions
                success = True
            except Exception:
                success = False

        assert success is True


class TestRegressionScenarios:
    """Test specific regression scenarios that would have failed before the fix."""

    def test_session_manager_workflow_simulation(self):
        """
        Simulate the session manager workflow that was failing.

        This simulates the process_initial_criteria workflow where
        criteria are added after session creation.
        """
        # Step 1: Create session (like session_manager.create_session)
        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Database Selection",
        )

        # Step 2: Add options (like in session creation)
        session.add_option("PostgreSQL", "Relational database")
        session.add_option("MongoDB", "Document database")

        # Step 3: Process initial criteria (this was failing before fix)
        # Simulate services/decision_service.py:process_initial_criteria
        criteria_specs = [
            {
                "name": "ACID Compliance",
                "description": "Transaction support",
                "weight": 2.0,
                "model_backend": ModelBackend.BEDROCK,
            },
            {
                "name": "Scalability",
                "description": "Horizontal scaling capability",
                "weight": 1.5,
                "model_backend": ModelBackend.LITELLM,
            },
        ]

        # Each criterion addition triggers cache invalidation
        for spec in criteria_specs:
            criterion = Criterion(
                name=spec["name"],
                description=spec["description"],
                weight=spec["weight"],
                model_backend=spec["model_backend"],
            )
            session.add_criterion(criterion)  # This would fail before the fix

        # Verify everything worked
        assert len(session.criteria) == 2
        assert len(session.options) == 2
        assert "decision_matrix" not in session.__dict__

        # Property access should work
        matrix = session.decision_matrix
        if "matrix" in matrix:
            assert len(matrix["matrix"]) == 2  # Two options
            assert len(matrix["criteria_weights"]) == 2  # Two criteria
        else:
            # Still in setup state without evaluations
            assert matrix["has_criteria"] is True
            assert len(session.options) == 2  # Two options
            assert len(session.criteria) == 2  # Two criteria

    def test_concurrent_criteria_addition(self):
        """Test that the fix works under concurrent criteria addition."""
        import threading
        import time

        session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Concurrent Test",
        )

        session.add_option("Option A")

        results = {"errors": [], "criteria_count": 0}

        def add_criteria_worker(worker_id):
            """Add criteria from worker thread"""
            try:
                for i in range(3):
                    criterion = Criterion(
                        name=f"Worker{worker_id}_Criterion{i}",
                        description=f"Criterion from worker {worker_id}",
                        weight=1.0 + i,
                    )
                    session.add_criterion(criterion)
                    results["criteria_count"] += 1
                    time.sleep(0.001)  # Small delay to increase race condition chance
            except Exception as e:
                results["errors"].append(f"Worker {worker_id}: {e!s}")

        # Create multiple threads adding criteria
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_criteria_worker, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have errors (the fix prevents race conditions in cache invalidation)
        assert len(results["errors"]) == 0
        assert results["criteria_count"] == 9  # 3 workers * 3 criteria each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
