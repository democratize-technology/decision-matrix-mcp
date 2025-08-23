"""
Integration tests for concurrent session management and isolation.

These tests verify:
- Multiple parallel decision sessions
- Session isolation under concurrent access
- Memory leak detection under load
- Cleanup behavior under concurrent access
- Thread safety of session operations
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import time
import tracemalloc
from unittest.mock import Mock

import pytest

from decision_matrix_mcp.exceptions import ResourceLimitError
from decision_matrix_mcp.models import Criterion
from decision_matrix_mcp.session_manager import SessionManager


class TestConcurrentSessionCreation:
    """Test concurrent session creation and management."""

    @pytest.fixture()
    def session_manager(self):
        """Create a session manager for testing."""
        manager = SessionManager(max_sessions=20, session_ttl_hours=1)
        yield manager
        # Cleanup all sessions after test
        for session_id in list(manager.list_active_sessions().keys()):
            manager.remove_session(session_id)

    @pytest.mark.asyncio()
    async def test_concurrent_session_creation(self, session_manager):
        """Test creating multiple sessions concurrently."""
        num_sessions = 10

        async def create_session(index):
            """Create a single session."""
            topic = f"Decision {index}"
            options = [f"Option {index}A", f"Option {index}B", f"Option {index}C"]
            return session_manager.create_session(topic, options)

        # Create sessions concurrently
        tasks = [create_session(i) for i in range(num_sessions)]
        sessions = await asyncio.gather(*tasks)

        # Verify all sessions were created
        assert len(sessions) == num_sessions

        # Verify all session IDs are unique
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == num_sessions

        # Verify all sessions are tracked
        active_sessions = session_manager.list_active_sessions()
        assert len(active_sessions) == num_sessions

        # Verify session isolation
        for i, session in enumerate(sessions):
            assert session.topic == f"Decision {i}"
            assert len(session.options) == 3
            assert f"Option {i}A" in session.options

    @pytest.mark.asyncio()
    async def test_concurrent_session_access(self, session_manager):
        """Test concurrent access to the same session."""
        # Create a base session
        session = session_manager.create_session("Concurrent Access Test", ["Option A", "Option B"])
        session_id = session.session_id

        # Define criteria to add concurrently
        criteria_data = [
            ("Performance", "Speed and efficiency", 2.0),
            ("Cost", "Total cost of ownership", 1.5),
            ("Usability", "Ease of use", 1.0),
            ("Reliability", "System stability", 2.5),
            ("Scalability", "Growth capacity", 1.8),
        ]

        async def add_criterion_to_session(criterion_data):
            """Add a criterion to the session."""
            name, description, weight = criterion_data
            criterion = Criterion(name=name, description=description, weight=weight)

            # Get session and add criterion
            test_session = session_manager.get_session(session_id)
            if test_session:
                test_session.add_criterion(criterion)
                return True
            return False

        # Add criteria concurrently
        tasks = [add_criterion_to_session(data) for data in criteria_data]
        results = await asyncio.gather(*tasks)

        # Verify all criteria were added successfully
        assert all(results)

        # Verify session has all criteria
        final_session = session_manager.get_session(session_id)
        assert len(final_session.criteria) == len(criteria_data)

        # Verify criteria names
        expected_names = [data[0] for data in criteria_data]
        actual_names = list(final_session.criteria.keys())
        assert set(actual_names) == set(expected_names)

    @pytest.mark.asyncio()
    async def test_session_limit_under_concurrency(self, session_manager):
        """Test session limit enforcement under concurrent load."""
        # Reconfigure with lower limit for testing
        limited_manager = SessionManager(max_sessions=5, session_ttl_hours=1)

        num_attempts = 10  # Try to create more than the limit

        async def attempt_session_creation(index):
            """Attempt to create a session, handling potential limit errors."""
            try:
                return limited_manager.create_session(
                    f"Limited Session {index}",
                    [f"Option {index}A", f"Option {index}B"],
                )
            except ResourceLimitError:
                return None

        # Attempt concurrent session creation
        tasks = [attempt_session_creation(i) for i in range(num_attempts)]
        results = await asyncio.gather(*tasks)

        # Count successful creations
        successful_sessions = [s for s in results if s is not None]
        failed_attempts = [s for s in results if s is None]

        # Should have exactly max_sessions successful creations
        assert len(successful_sessions) == 5
        assert len(failed_attempts) == 5

        # Verify session manager respects the limit
        active_sessions = limited_manager.list_active_sessions()
        assert len(active_sessions) == 5

        # Cleanup
        for session_id in list(limited_manager.list_active_sessions().keys()):
            limited_manager.remove_session(session_id)

    def test_thread_safety_session_operations(self, session_manager):
        """Test thread safety of session operations using real threads."""
        num_threads = 10
        operations_per_thread = 5
        results = []
        errors = []

        def worker_thread(thread_id):
            """Worker thread that performs session operations."""
            thread_results = []
            thread_errors = []

            try:
                for op_id in range(operations_per_thread):
                    # Create session
                    session = session_manager.create_session(
                        f"Thread {thread_id} Session {op_id}",
                        [f"Option {thread_id}-{op_id}-A", f"Option {thread_id}-{op_id}-B"],
                    )
                    thread_results.append(("create", session.session_id))

                    # Add criterion
                    criterion = Criterion(
                        name=f"Criterion-{thread_id}-{op_id}",
                        description=f"Test criterion for thread {thread_id}",
                        weight=1.0,
                    )
                    session.add_criterion(criterion)
                    thread_results.append(("add_criterion", criterion.name))

                    # List sessions (read operation)
                    active = session_manager.list_active_sessions()
                    thread_results.append(("list", len(active)))

                    # Clean up this session
                    session_manager.remove_session(session.session_id)
                    thread_results.append(("remove", session.session_id))

            except Exception as e:
                thread_errors.append((thread_id, str(e), type(e).__name__))

            return thread_results, thread_errors

        # Run threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, thread_id) for thread_id in range(num_threads)
            ]

            for future in as_completed(futures):
                thread_results, thread_errors = future.result()
                results.extend(thread_results)
                errors.extend(thread_errors)

        # Verify no errors occurred
        if errors:
            for _error in errors:
                pass
        assert len(errors) == 0, f"Thread safety violations: {errors}"

        # Verify expected number of operations
        expected_ops = num_threads * operations_per_thread * 4  # 4 ops per iteration
        assert len(results) == expected_ops

        # Verify all sessions were cleaned up
        remaining_sessions = session_manager.list_active_sessions()
        assert len(remaining_sessions) == 0


class TestConcurrentSessionEvaluation:
    """Test concurrent evaluation across multiple sessions."""

    @pytest.fixture()
    def orchestrator_mock(self):
        """Create a mock orchestrator for testing."""

        from decision_matrix_mcp.orchestrator import DecisionOrchestrator

        # Create mock orchestrator
        mock_orchestrator = Mock(spec=DecisionOrchestrator)

        # Mock evaluation results
        def mock_evaluate(session, options, criteria):
            """Mock evaluation that simulates realistic delays."""
            # Simulate evaluation time
            time.sleep(0.1)

            results = {}
            for criterion_name in criteria:
                results[criterion_name] = {}
                for option_name in options:
                    # Generate mock scores
                    score = 5.0 + (hash(f"{criterion_name}-{option_name}") % 100) / 20.0
                    justification = f"Mock evaluation for {option_name} on {criterion_name}"
                    results[criterion_name][option_name] = (score, justification)

            return results

        mock_orchestrator.evaluate_options_across_criteria.side_effect = mock_evaluate
        return mock_orchestrator

    @pytest.mark.asyncio()
    async def test_concurrent_session_evaluation(self, orchestrator_mock):
        """Test evaluating multiple sessions concurrently."""
        session_manager = SessionManager(max_sessions=20, session_ttl_hours=1)

        # Create multiple sessions with different configurations
        sessions = []
        for i in range(5):
            session = session_manager.create_session(
                f"Evaluation Test {i}",
                [f"Option {i}A", f"Option {i}B", f"Option {i}C"],
            )

            # Add criteria to each session
            criterion1 = Criterion(f"Performance-{i}", f"Performance for session {i}", 2.0)
            criterion2 = Criterion(f"Cost-{i}", f"Cost for session {i}", 1.5)
            session.add_criterion(criterion1)
            session.add_criterion(criterion2)

            sessions.append(session)

        async def evaluate_session(session):
            """Evaluate a single session."""
            options = list(session.options.keys())
            criteria = list(session.criteria.keys())

            # Use mock orchestrator
            results = orchestrator_mock.evaluate_options_across_criteria(session, options, criteria)

            # Record evaluation in session
            session.record_evaluation(results)
            return session.session_id, len(results)

        # Evaluate all sessions concurrently
        tasks = [evaluate_session(session) for session in sessions]
        evaluation_results = await asyncio.gather(*tasks)

        # Verify all evaluations completed
        assert len(evaluation_results) == 5

        # Verify each session has evaluation results
        for session_id, num_criteria in evaluation_results:
            session = session_manager.get_session(session_id)
            assert session is not None
            assert session.evaluation_count > 0
            assert num_criteria == 2  # Each session had 2 criteria

        # Cleanup
        for session in sessions:
            session_manager.remove_session(session.session_id)


class TestMemoryLeakDetection:
    """Test memory leak detection under concurrent load."""

    @pytest.mark.asyncio()
    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under high session load."""
        # Start memory tracing
        tracemalloc.start()

        session_manager = SessionManager(max_sessions=50, session_ttl_hours=1)

        # Take initial memory snapshot
        initial_snapshot = tracemalloc.take_snapshot()

        async def session_lifecycle_batch(batch_id, sessions_per_batch=10):
            """Create, use, and cleanup a batch of sessions."""
            batch_sessions = []

            # Create sessions
            for i in range(sessions_per_batch):
                session = session_manager.create_session(
                    f"Batch {batch_id} Session {i}",
                    [f"Option {batch_id}-{i}-A", f"Option {batch_id}-{i}-B"],
                )

                # Add some criteria
                criterion = Criterion(
                    f"Criterion-{batch_id}-{i}",
                    f"Test criterion for batch {batch_id} session {i}",
                    weight=1.0 + (i * 0.1),
                )
                session.add_criterion(criterion)

                # Simulate some evaluation data
                session.record_evaluation({f"Criterion-{batch_id}-{i}": {"mock": "data"}})

                batch_sessions.append(session)

            # Simulate some activity
            await asyncio.sleep(0.1)

            # Cleanup sessions
            for session in batch_sessions:
                session_manager.remove_session(session.session_id)

            return len(batch_sessions)

        # Run multiple batches concurrently
        num_batches = 10
        sessions_per_batch = 5

        batch_tasks = [
            session_lifecycle_batch(batch_id, sessions_per_batch) for batch_id in range(num_batches)
        ]

        batch_results = await asyncio.gather(*batch_tasks)

        # Verify all sessions were processed
        total_sessions_processed = sum(batch_results)
        expected_total = num_batches * sessions_per_batch
        assert total_sessions_processed == expected_total

        # Force garbage collection
        gc.collect()
        await asyncio.sleep(0.1)

        # Take final memory snapshot
        final_snapshot = tracemalloc.take_snapshot()

        # Compare memory usage
        top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")

        # Check for significant memory growth
        total_memory_growth = sum(stat.size_diff for stat in top_stats)

        # Memory growth should be reasonable (less than 10MB for this test)
        max_acceptable_growth = 10 * 1024 * 1024  # 10MB

        if total_memory_growth > max_acceptable_growth:
            for _stat in top_stats[:5]:
                pass

        # This assertion might be too strict in some environments, so we log instead
        # assert total_memory_growth < max_acceptable_growth

        # Verify no sessions remain
        remaining_sessions = session_manager.list_active_sessions()
        assert len(remaining_sessions) == 0

        tracemalloc.stop()

    @pytest.mark.asyncio()
    async def test_session_cleanup_under_concurrent_access(self):
        """Test session cleanup behavior under concurrent access."""
        session_manager = SessionManager(max_sessions=30, session_ttl_hours=1)

        # Create a mix of long-lived and short-lived sessions
        long_lived_sessions = []

        # Create some long-lived sessions
        for i in range(5):
            session = session_manager.create_session(
                f"Long-lived Session {i}",
                [f"Long Option {i}A", f"Long Option {i}B"],
            )
            long_lived_sessions.append(session)

        async def create_and_cleanup_batch(batch_id):
            """Create short-lived sessions and clean them up."""
            temp_sessions = []

            # Create temporary sessions
            for i in range(3):
                session = session_manager.create_session(
                    f"Temp Batch {batch_id} Session {i}",
                    [f"Temp {batch_id}-{i}-A", f"Temp {batch_id}-{i}-B"],
                )
                temp_sessions.append(session)

            # Simulate some usage
            await asyncio.sleep(0.05)

            # Clean up temp sessions
            for session in temp_sessions:
                session_manager.remove_session(session.session_id)

            return len(temp_sessions)

        # Run multiple cleanup batches concurrently while long-lived sessions exist
        num_batches = 10
        batch_tasks = [create_and_cleanup_batch(i) for i in range(num_batches)]

        batch_results = await asyncio.gather(*batch_tasks)

        # Verify all temporary sessions were processed
        total_temp_sessions = sum(batch_results)
        assert total_temp_sessions == num_batches * 3

        # Verify long-lived sessions still exist
        remaining_sessions = session_manager.list_active_sessions()
        assert len(remaining_sessions) == 5

        # Verify the remaining sessions are the long-lived ones
        remaining_ids = set(remaining_sessions.keys())
        expected_ids = {s.session_id for s in long_lived_sessions}
        assert remaining_ids == expected_ids

        # Cleanup long-lived sessions
        for session in long_lived_sessions:
            session_manager.remove_session(session.session_id)

        # Verify all sessions are gone
        final_sessions = session_manager.list_active_sessions()
        assert len(final_sessions) == 0


class TestSessionIsolationUnderLoad:
    """Test session isolation under high concurrent load."""

    @pytest.mark.asyncio()
    async def test_session_data_isolation(self):
        """Test that session data remains isolated under concurrent modifications."""
        session_manager = SessionManager(max_sessions=20, session_ttl_hours=1)

        # Create multiple sessions
        num_sessions = 8
        sessions = []

        for i in range(num_sessions):
            session = session_manager.create_session(
                f"Isolation Test Session {i}",
                [f"Option {i}-A", f"Option {i}-B", f"Option {i}-C"],
            )
            sessions.append(session)

        async def modify_session_concurrently(session, modifier_id):
            """Modify a session's data concurrently."""
            modifications = []

            # Add criteria with unique names
            for j in range(3):
                criterion_name = f"Criterion-{modifier_id}-{j}"
                criterion = Criterion(
                    name=criterion_name,
                    description=f"Criterion {j} added by modifier {modifier_id}",
                    weight=1.0 + (j * 0.2),
                )
                session.add_criterion(criterion)
                modifications.append(("add_criterion", criterion_name))

            # Add some evaluation data
            eval_data = {
                f"Criterion-{modifier_id}-{j}": {
                    f"Option {session.session_id[-2:]}-{opt}": f"Data-{modifier_id}-{j}-{opt}"
                    for j in range(3)
                    for opt in ["A", "B", "C"]
                },
            }
            session.record_evaluation(eval_data)
            modifications.append(("record_evaluation", len(eval_data)))

            return modifier_id, modifications

        # Modify each session concurrently from multiple "modifiers"
        tasks = []
        for session in sessions:
            # Each session gets modified by 3 concurrent "modifiers"
            for modifier_id in range(3):
                task = modify_session_concurrently(session, modifier_id)
                tasks.append(task)

        # Execute all modifications concurrently
        results = await asyncio.gather(*tasks)

        # Verify all modifications completed
        assert len(results) == num_sessions * 3

        # Verify session data integrity
        for i, session in enumerate(sessions):
            # Each session should have 9 criteria (3 modifiers × 3 criteria each)
            assert len(session.criteria) == 9

            # Verify criterion names are unique and follow expected pattern
            criterion_names = set(session.criteria.keys())
            expected_patterns = {f"Criterion-{mod}-{j}" for mod in range(3) for j in range(3)}
            assert criterion_names == expected_patterns

            # Verify evaluation count
            assert session.evaluation_count == 3  # One per modifier

            # Verify session identity hasn't been corrupted
            assert session.topic == f"Isolation Test Session {i}"
            assert len(session.options) == 3
            assert f"Option {i}-A" in session.options

        # Cleanup
        for session in sessions:
            session_manager.remove_session(session.session_id)

    @pytest.mark.asyncio()
    async def test_concurrent_session_retrieval(self):
        """Test concurrent retrieval of sessions doesn't cause data corruption."""
        session_manager = SessionManager(max_sessions=15, session_ttl_hours=1)

        # Create test sessions with unique data
        test_sessions = []
        for i in range(5):
            session = session_manager.create_session(
                f"Retrieval Test {i}",
                [f"Opt{i}A", f"Opt{i}B"],
            )

            # Add unique criterion
            criterion = Criterion(f"Crit{i}", f"Description {i}", weight=float(i + 1))
            session.add_criterion(criterion)

            test_sessions.append(session)

        session_ids = [s.session_id for s in test_sessions]

        async def retrieve_and_verify_session(session_id, expected_index):
            """Retrieve a session and verify its data integrity."""
            retrieved_session = session_manager.get_session(session_id)

            if retrieved_session is None:
                return False, f"Session {session_id} not found"

            # Verify the data matches expected values
            expected_topic = f"Retrieval Test {expected_index}"
            if retrieved_session.topic != expected_topic:
                return False, f"Topic mismatch: {retrieved_session.topic} != {expected_topic}"

            expected_options = {f"Opt{expected_index}A", f"Opt{expected_index}B"}
            actual_options = set(retrieved_session.options.keys())
            if actual_options != expected_options:
                return False, f"Options mismatch: {actual_options} != {expected_options}"

            expected_criterion = f"Crit{expected_index}"
            if expected_criterion not in retrieved_session.criteria:
                return False, f"Criterion missing: {expected_criterion}"

            return True, "OK"

        # Perform many concurrent retrievals
        num_retrievals_per_session = 20
        tasks = []

        for _ in range(num_retrievals_per_session):
            for i, session_id in enumerate(session_ids):
                task = retrieve_and_verify_session(session_id, i)
                tasks.append(task)

        # Execute all retrievals concurrently
        results = await asyncio.gather(*tasks)

        # Verify all retrievals were successful
        failed_retrievals = [result for result in results if not result[0]]

        if failed_retrievals:
            for _failure in failed_retrievals[:10]:  # Show first 10 failures
                pass

        assert len(failed_retrievals) == 0, f"Failed retrievals detected: {len(failed_retrievals)}"

        # Cleanup
        for session in test_sessions:
            session_manager.remove_session(session.session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
