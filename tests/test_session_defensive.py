"""
Defensive programming tests for session management edge cases.

Tests session manager behavior under race conditions, TTL edge cases,
and concurrent access patterns that could expose defensive code branches.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from unittest.mock import patch

from decision_matrix_mcp.exceptions import ResourceLimitError, SessionNotFoundError
from decision_matrix_mcp.models import Criterion
from decision_matrix_mcp.session_manager import SessionManager


class TestSessionManagerDefensive:
    """Test defensive patterns in session management."""

    def test_session_limit_race_condition_defensive(self):
        """Test session limit enforcement under race conditions."""
        manager = SessionManager(max_sessions=2, session_ttl_hours=1)

        # Simulate race condition where multiple threads check limit simultaneously
        barrier = threading.Barrier(3)  # 3 threads
        results = []
        errors = []

        def create_session_with_race(thread_id):
            """Create session after barrier to simulate simultaneous access."""
            try:
                barrier.wait()  # All threads start at same time
                session = manager.create_session(f"Race Session {thread_id}", ["A", "B"])
                results.append(("success", thread_id, session.session_id))
            except ResourceLimitError as e:
                errors.append(("limit_error", thread_id, str(e)))
            except Exception as e:
                errors.append(("other_error", thread_id, str(e)))

        # Start 3 threads simultaneously (limit is 2)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_session_with_race, i) for i in range(3)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Exactly 2 should succeed, 1 should fail due to limit
        assert len(results) == 2, f"Expected 2 successes, got {len(results)}"
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"
        assert errors[0][0] == "limit_error"

        # Verify session manager state is consistent
        active_sessions = manager.list_active_sessions()
        assert len(active_sessions) == 2

    def test_ttl_cleanup_race_conditions(self):
        """Test TTL cleanup under concurrent session modifications."""
        manager = SessionManager(max_sessions=10, session_ttl_hours=0.001)  # 3.6 second TTL

        # Create sessions that will expire
        sessions = []
        for i in range(5):
            session = manager.create_session(f"TTL Session {i}", ["A", "B"])
            sessions.append(session)

        # Wait for TTL expiration
        time.sleep(0.005)  # 18 seconds (5x TTL)

        # Concurrently access expired sessions while cleanup runs
        def access_expired_session(session_id):
            try:
                return manager.get_session(session_id)
            except SessionNotFoundError:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_expired_session, s.session_id) for s in sessions]
            results = [f.result() for f in as_completed(futures)]

        # All should be None due to TTL expiration
        assert all(r is None for r in results)

        # Session manager should have cleaned up
        active_sessions = manager.list_active_sessions()
        assert len(active_sessions) == 0

    def test_concurrent_session_modification_defensive(self):
        """Test concurrent modification of session data with defensive patterns."""
        manager = SessionManager(max_sessions=10, session_ttl_hours=1)
        session = manager.create_session("Concurrent Test", ["A", "B", "C"])
        session_id = session.session_id

        # Define criteria to add concurrently
        criteria_data = [(f"Criterion-{i}", f"Description {i}", i * 0.5 + 1.0) for i in range(20)]

        def add_criterion_safely(criterion_data):
            """Add criterion with defensive error handling."""
            name, description, weight = criterion_data
            try:
                test_session = manager.get_session(session_id)
                if test_session:
                    criterion = Criterion(name=name, description=description, weight=weight)
                    test_session.add_criterion(criterion)
                    return ("success", name)
            except Exception as e:
                return ("error", name, str(e))
            return ("session_not_found", name)

        # Add criteria concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_criterion_safely, data) for data in criteria_data]
            results = [f.result() for f in as_completed(futures)]

        # Verify all operations succeeded
        successes = [r for r in results if r[0] == "success"]
        assert len(successes) == 20, f"Expected 20 successes, got {len(successes)}"

        # Verify session integrity
        final_session = manager.get_session(session_id)
        assert len(final_session.criteria) == 20

        # Verify no duplicate criteria names
        criterion_names = list(final_session.criteria.keys())
        assert len(set(criterion_names)) == 20  # All unique

    def test_session_cleanup_with_partial_failures(self):
        """Test session cleanup when some cleanup operations fail."""
        manager = SessionManager(max_sessions=5, session_ttl_hours=1)

        # Create sessions
        sessions = []
        for i in range(5):
            session = manager.create_session(f"Cleanup Test {i}", ["A", "B"])
            sessions.append(session)

        # Mock partial cleanup failures
        original_remove = manager.remove_session

        def failing_remove(session_id):
            # Fail for session 2, succeed for others
            if "Test 2" in manager.get_session(session_id).topic:
                raise Exception("Simulated cleanup failure")
            return original_remove(session_id)

        with patch.object(manager, "remove_session", side_effect=failing_remove):
            # Attempt to clean up all sessions
            cleanup_results = []
            for session in sessions:
                try:
                    manager.remove_session(session.session_id)
                    cleanup_results.append(("success", session.session_id))
                except Exception as e:
                    cleanup_results.append(("failed", session.session_id, str(e)))

        # Should have 4 successes, 1 failure
        successes = [r for r in cleanup_results if r[0] == "success"]
        failures = [r for r in cleanup_results if r[0] == "failed"]
        assert len(successes) == 4
        assert len(failures) == 1

        # Failed session should still exist in active sessions
        active_sessions = manager.list_active_sessions()
        assert len(active_sessions) == 1

    def test_session_access_during_cleanup_defensive(self):
        """Test accessing sessions during cleanup operations."""
        manager = SessionManager(max_sessions=10, session_ttl_hours=0.001)  # Short TTL

        # Create sessions that will expire
        sessions = []
        for i in range(3):
            session = manager.create_session(f"Access During Cleanup {i}", ["A", "B"])
            sessions.append(session)

        # Wait for expiration
        time.sleep(0.005)

        # Mock cleanup to be slow
        original_cleanup = manager._cleanup_expired_sessions

        def slow_cleanup():
            time.sleep(0.01)  # Slow cleanup
            return original_cleanup()

        cleanup_started = threading.Event()

        def trigger_cleanup():
            cleanup_started.set()
            return slow_cleanup()

        # Start cleanup in background
        cleanup_thread = threading.Thread(target=trigger_cleanup)
        cleanup_thread.start()

        # Wait for cleanup to start
        cleanup_started.wait()

        # Try to access sessions during cleanup
        access_results = []
        for session in sessions:
            try:
                result = manager.get_session(session.session_id)
                access_results.append(("found" if result else "not_found", session.session_id))
            except SessionNotFoundError:
                access_results.append(("not_found_exception", session.session_id))

        cleanup_thread.join()

        # All accesses should handle cleanup gracefully (no crashes)
        assert len(access_results) == 3
        # Results should be consistent (either all not found or gracefully handled)
        assert all(r[0] in ["found", "not_found", "not_found_exception"] for r in access_results)

    def test_memory_leak_prevention_defensive(self):
        """Test that session manager prevents memory leaks under stress."""
        manager = SessionManager(max_sessions=50, session_ttl_hours=0.001)  # Very short TTL

        # Create and expire many sessions rapidly
        created_sessions = []
        for batch in range(5):  # 5 batches of 10 sessions each
            batch_sessions = []
            for i in range(10):
                session = manager.create_session(f"Batch {batch} Session {i}", ["A", "B"])
                # Add some criteria to make sessions heavier
                for j in range(3):
                    criterion = Criterion(f"Criterion-{j}", f"Description {j}", j + 1.0)
                    session.add_criterion(criterion)
                batch_sessions.append(session)

            created_sessions.extend(batch_sessions)

            # Wait for TTL expiration
            time.sleep(0.005)

            # Trigger cleanup
            manager._cleanup_expired_sessions()

            # Verify cleanup worked
            active_sessions = manager.list_active_sessions()
            assert (
                len(active_sessions) == 0
            ), f"Batch {batch}: Expected 0 active sessions, got {len(active_sessions)}"

        # Final verification - no sessions should remain
        final_active = manager.list_active_sessions()
        assert len(final_active) == 0

    def test_session_data_integrity_under_concurrent_access(self):
        """Test session data integrity with high concurrent access."""
        manager = SessionManager(max_sessions=20, session_ttl_hours=1)
        session = manager.create_session(
            "Integrity Test", ["OptionA", "OptionB", "OptionC", "OptionD"]
        )
        session_id = session.session_id

        # Concurrent operations: reads, writes, modifications
        def concurrent_operations(worker_id):
            results = []
            try:
                for op_id in range(10):  # 10 operations per worker
                    # Mix of operations
                    if op_id % 3 == 0:  # Read operation
                        session_data = manager.get_session(session_id)
                        if session_data:
                            results.append(("read", len(session_data.criteria)))
                        else:
                            results.append(("read_failed", None))

                    elif op_id % 3 == 1:  # Write operation (add criterion)
                        session_data = manager.get_session(session_id)
                        if session_data:
                            criterion = Criterion(
                                name=f"Worker-{worker_id}-Criterion-{op_id}",
                                description=f"From worker {worker_id}",
                                weight=1.0,
                            )
                            session_data.add_criterion(criterion)
                            results.append(("write", criterion.name))
                        else:
                            results.append(("write_failed", None))

                    else:  # List operations
                        active = manager.list_active_sessions()
                        results.append(("list", len(active)))

                    # Small delay to interleave operations
                    time.sleep(0.001)

            except Exception as e:
                results.append(("error", str(e)))

            return results

        # Run with 8 concurrent workers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(concurrent_operations, i) for i in range(8)]
            all_results = [f.result() for f in as_completed(futures)]

        # Verify no errors occurred
        all_operations = [op for worker_results in all_results for op in worker_results]
        errors = [op for op in all_operations if op[0] == "error"]
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"

        # Verify session integrity
        final_session = manager.get_session(session_id)
        assert final_session is not None
        assert len(final_session.options) == 4  # Original options preserved

        # Count successful writes
        writes = [op for op in all_operations if op[0] == "write"]
        expected_criteria_count = len(writes)
        actual_criteria_count = len(final_session.criteria)

        # Should have all written criteria (no lost writes)
        assert (
            actual_criteria_count == expected_criteria_count
        ), f"Expected {expected_criteria_count} criteria, got {actual_criteria_count}"

        # All criteria names should be unique (no duplicates)
        criterion_names = list(final_session.criteria.keys())
        assert len(set(criterion_names)) == len(criterion_names), "Duplicate criteria detected"
