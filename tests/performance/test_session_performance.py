"""
Performance tests for session management operations.

These tests measure:
- Session creation/cleanup performance
- Memory growth patterns during session lifecycle
- Cleanup efficiency testing
- LRU eviction performance
- Session lookup and retrieval performance
"""

import asyncio
import gc
from statistics import mean, median, stdev
import time
import tracemalloc

import pytest

from decision_matrix_mcp.models import Criterion, Score
from decision_matrix_mcp.session_manager import SessionManager


class SessionPerformanceProfiler:
    """Profiler for session management performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.operation_times = {}
        self.memory_snapshots = {}
        self.session_counts = {}

    def time_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Time a session operation."""
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = end_time - start_time

        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []

        self.operation_times[operation_name].append(duration)
        return result, duration

    async def time_async_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Time an async session operation."""
        start_time = time.perf_counter()
        result = await operation_func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = end_time - start_time

        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []

        self.operation_times[operation_name].append(duration)
        return result, duration

    def take_memory_snapshot(self, snapshot_name: str):
        """Take a memory snapshot."""
        gc.collect()
        if tracemalloc.is_tracing():
            self.memory_snapshots[snapshot_name] = tracemalloc.take_snapshot()

    def get_memory_diff(self, before_snapshot: str, after_snapshot: str) -> int:
        """Get memory difference between two snapshots in bytes."""
        if before_snapshot in self.memory_snapshots and after_snapshot in self.memory_snapshots:
            before = self.memory_snapshots[before_snapshot]
            after = self.memory_snapshots[after_snapshot]
            diff = after.compare_to(before, "lineno")
            return sum(stat.size_diff for stat in diff)
        return 0

    def get_operation_stats(self, operation_name: str) -> dict:
        """Get statistics for an operation."""
        if operation_name not in self.operation_times:
            return {}

        times = self.operation_times[operation_name]
        if not times:
            return {}

        return {
            "count": len(times),
            "mean": mean(times),
            "median": median(times),
            "min": min(times),
            "max": max(times),
            "std": stdev(times) if len(times) > 1 else 0.0,
            "total": sum(times),
        }

    def print_summary(self):
        """Print performance summary."""

        for operation_name in sorted(self.operation_times.keys()):
            stats = self.get_operation_stats(operation_name)
            if stats:
                pass


class TestSessionCreationPerformance:
    """Test session creation and initialization performance."""

    @pytest.fixture()
    def profiler(self):
        """Create performance profiler."""
        return SessionPerformanceProfiler()

    @pytest.mark.parametrize("num_sessions", [10, 50, 100, 200])
    @pytest.mark.asyncio()
    async def test_session_creation_scaling(self, profiler, num_sessions):
        """Test session creation performance at different scales."""
        session_manager = SessionManager(max_sessions=num_sessions + 50, session_ttl_hours=1)

        tracemalloc.start()
        profiler.take_memory_snapshot("before_creation")

        created_sessions = []
        creation_times = []

        # Create sessions sequentially to measure individual creation time
        for i in range(num_sessions):
            options = [f"Session{i}Opt{j}" for j in range(3)]

            _, creation_time = profiler.time_operation(
                "session_creation",
                session_manager.create_session,
                f"Performance Test Session {i}",
                options,
            )

            created_sessions.append(_)
            creation_times.append(creation_time)

        profiler.take_memory_snapshot("after_creation")

        # Measure retrieval performance
        retrieval_times = []
        for session in created_sessions[: min(50, len(created_sessions))]:  # Test first 50
            _, retrieval_time = profiler.time_operation(
                "session_retrieval",
                session_manager.get_session,
                session.session_id,
            )
            retrieval_times.append(retrieval_time)

        # Measure listing performance
        _, listing_time = profiler.time_operation(
            "session_listing",
            session_manager.list_active_sessions,
        )

        # Calculate memory usage
        memory_growth = profiler.get_memory_diff("before_creation", "after_creation")
        memory_per_session = memory_growth / num_sessions if num_sessions > 0 else 0

        # Performance analysis
        mean_creation_time = mean(creation_times)
        mean_retrieval_time = mean(retrieval_times)

        # Performance assertions
        max_creation_time = 0.010  # 10ms max per session
        assert (
            mean_creation_time < max_creation_time
        ), f"Session creation too slow: {mean_creation_time * 1000:.2f}ms > {max_creation_time * 1000:.2f}ms"

        max_retrieval_time = 0.001  # 1ms max per retrieval
        assert (
            mean_retrieval_time < max_retrieval_time
        ), f"Session retrieval too slow: {mean_retrieval_time * 1000:.2f}ms > {max_retrieval_time * 1000:.2f}ms"

        max_listing_time = 0.005  # 5ms max for listing
        assert (
            listing_time < max_listing_time
        ), f"Session listing too slow: {listing_time * 1000:.2f}ms > {max_listing_time * 1000:.2f}ms"

        max_memory_per_session = 50 * 1024  # 50KB max per session
        if memory_per_session > max_memory_per_session:
            pass

        # Cleanup
        for session in created_sessions:
            session_manager.remove_session(session.session_id)

        tracemalloc.stop()

    @pytest.mark.asyncio()
    async def test_concurrent_session_creation_performance(self, profiler):
        """Test concurrent session creation performance."""
        session_manager = SessionManager(max_sessions=100, session_ttl_hours=1)
        num_concurrent = 20

        tracemalloc.start()
        profiler.take_memory_snapshot("before_concurrent")

        async def create_session_batch(batch_id):
            """Create a session asynchronously."""
            options = [f"Batch{batch_id}Opt{j}" for j in range(4)]
            return session_manager.create_session(f"Concurrent Session {batch_id}", options)

        # Measure concurrent creation time
        start_time = time.perf_counter()

        # Create sessions concurrently
        tasks = [create_session_batch(i) for i in range(num_concurrent)]
        concurrent_sessions = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        concurrent_creation_time = end_time - start_time

        profiler.take_memory_snapshot("after_concurrent")

        # Verify all sessions were created
        assert len(concurrent_sessions) == num_concurrent
        assert len({s.session_id for s in concurrent_sessions}) == num_concurrent

        # Measure concurrent retrieval
        async def retrieve_session(session_id):
            """Retrieve a session asynchronously."""
            return session_manager.get_session(session_id)

        start_time = time.perf_counter()
        retrieval_tasks = [retrieve_session(s.session_id) for s in concurrent_sessions]
        retrieved_sessions = await asyncio.gather(*retrieval_tasks)
        end_time = time.perf_counter()
        concurrent_retrieval_time = end_time - start_time

        # Verify all retrievals successful
        assert all(s is not None for s in retrieved_sessions)

        # Calculate metrics
        profiler.get_memory_diff("before_concurrent", "after_concurrent")
        creation_rate = num_concurrent / concurrent_creation_time
        retrieval_rate = num_concurrent / concurrent_retrieval_time

        # Performance assertions
        min_creation_rate = 50  # At least 50 sessions/sec
        assert (
            creation_rate > min_creation_rate
        ), f"Concurrent creation rate too low: {creation_rate:.1f} < {min_creation_rate}"

        min_retrieval_rate = 500  # At least 500 retrievals/sec
        assert (
            retrieval_rate > min_retrieval_rate
        ), f"Concurrent retrieval rate too low: {retrieval_rate:.1f} < {min_retrieval_rate}"

        # Cleanup
        for session in concurrent_sessions:
            session_manager.remove_session(session.session_id)

        tracemalloc.stop()


class TestSessionCleanupPerformance:
    """Test session cleanup and memory management performance."""

    @pytest.mark.asyncio()
    async def test_session_cleanup_performance(self):
        """Test performance of session cleanup operations."""
        session_manager = SessionManager(max_sessions=200, session_ttl_hours=1)
        profiler = SessionPerformanceProfiler()

        # Create a large number of sessions
        num_sessions = 100
        sessions = []

        tracemalloc.start()
        profiler.take_memory_snapshot("before_cleanup_test")

        # Create sessions with varying complexity
        for i in range(num_sessions):
            options = [f"CleanupOpt{i}_{j}" for j in range(3 + (i % 5))]  # Varying option counts
            session = session_manager.create_session(f"Cleanup Test {i}", options)

            # Add varying numbers of criteria
            criteria_count = 2 + (i % 4)
            for j in range(criteria_count):
                criterion = Criterion(
                    name=f"CleanupCrit{i}_{j}",
                    description=f"Cleanup criterion {j} for session {i}",
                    weight=1.0 + (j * 0.1),
                )
                session.add_criterion(criterion)

            # Add some evaluation data to make sessions more realistic
            if i % 3 == 0:
                eval_data = {
                    f"CleanupCrit{i}_{j}": {"mock": f"data_{i}_{j}"} for j in range(criteria_count)
                }
                session.record_evaluation(eval_data)

            sessions.append(session)

        profiler.take_memory_snapshot("after_creation_for_cleanup")

        # Test individual session removal performance
        removal_times = []
        sessions_to_remove = sessions[:50]  # Remove first 50 individually

        for session in sessions_to_remove:
            _, removal_time = profiler.time_operation(
                "individual_removal",
                session_manager.remove_session,
                session.session_id,
            )
            removal_times.append(removal_time)

        profiler.take_memory_snapshot("after_individual_removal")

        # Test batch cleanup performance
        remaining_sessions = sessions[50:]
        remaining_ids = [s.session_id for s in remaining_sessions]

        start_time = time.perf_counter()
        for session_id in remaining_ids:
            session_manager.remove_session(session_id)
        end_time = time.perf_counter()

        batch_cleanup_time = end_time - start_time

        profiler.take_memory_snapshot("after_batch_removal")

        # Force garbage collection and measure memory recovery
        gc.collect()
        profiler.take_memory_snapshot("after_gc")

        # Calculate metrics
        mean_removal_time = mean(removal_times)
        batch_removal_rate = len(remaining_sessions) / batch_cleanup_time

        memory_after_creation = profiler.get_memory_diff(
            "before_cleanup_test",
            "after_creation_for_cleanup",
        )
        profiler.get_memory_diff(
            "before_cleanup_test",
            "after_individual_removal",
        )
        profiler.get_memory_diff("before_cleanup_test", "after_batch_removal")
        memory_after_gc = profiler.get_memory_diff("before_cleanup_test", "after_gc")

        # Performance assertions
        max_removal_time = 0.005  # 5ms max per removal
        assert (
            mean_removal_time < max_removal_time
        ), f"Session removal too slow: {mean_removal_time * 1000:.2f}ms > {max_removal_time * 1000:.2f}ms"

        min_batch_rate = 100  # At least 100 removals/sec
        assert (
            batch_removal_rate > min_batch_rate
        ), f"Batch removal rate too low: {batch_removal_rate:.1f} < {min_batch_rate}"

        # Memory should be substantially recovered after cleanup and GC
        memory_recovery_ratio = (memory_after_creation - memory_after_gc) / memory_after_creation
        min_recovery_ratio = 0.8  # At least 80% memory recovery
        assert (
            memory_recovery_ratio > min_recovery_ratio
        ), f"Poor memory recovery: {memory_recovery_ratio:.2f} < {min_recovery_ratio}"

        # Verify all sessions are gone
        remaining = session_manager.list_active_sessions()
        assert len(remaining) == 0, f"Sessions not properly cleaned up: {len(remaining)} remaining"

        tracemalloc.stop()

    @pytest.mark.asyncio()
    async def test_session_ttl_cleanup_performance(self):
        """Test performance of TTL-based session cleanup."""
        # Use very short TTL for testing
        session_manager = SessionManager(max_sessions=50, session_ttl_hours=0.001)  # ~3.6 seconds
        profiler = SessionPerformanceProfiler()

        tracemalloc.start()
        profiler.take_memory_snapshot("before_ttl_test")

        # Create sessions that will expire
        expired_sessions = []
        for i in range(20):
            session = session_manager.create_session(
                f"TTL Test {i}",
                [f"TTLOpt{i}A", f"TTLOpt{i}B"],
            )
            expired_sessions.append(session)

        profiler.take_memory_snapshot("after_ttl_creation")

        # Wait for sessions to expire
        await asyncio.sleep(4)  # Wait longer than TTL

        # Trigger cleanup by creating a new session
        _, cleanup_time = profiler.time_operation(
            "ttl_cleanup",
            session_manager.create_session,
            "Trigger Cleanup",
            ["TriggerA", "TriggerB"],
        )

        profiler.take_memory_snapshot("after_ttl_cleanup")

        # Force garbage collection
        gc.collect()
        profiler.take_memory_snapshot("after_ttl_gc")

        # Verify expired sessions were cleaned up
        active_sessions = session_manager.list_active_sessions()

        # Should only have the trigger session
        assert (
            len(active_sessions) == 1
        ), f"TTL cleanup failed: {len(active_sessions)} sessions remain"

        # Calculate memory metrics
        profiler.get_memory_diff("before_ttl_test", "after_ttl_creation")
        profiler.get_memory_diff("before_ttl_test", "after_ttl_cleanup")
        profiler.get_memory_diff("before_ttl_test", "after_ttl_gc")

        # TTL cleanup should be reasonably fast
        max_ttl_cleanup_time = 0.050  # 50ms max for TTL cleanup
        assert (
            cleanup_time < max_ttl_cleanup_time
        ), f"TTL cleanup too slow: {cleanup_time * 1000:.2f}ms > {max_ttl_cleanup_time * 1000:.2f}ms"

        # Cleanup trigger session
        trigger_sessions = list(active_sessions.values())
        if trigger_sessions:
            session_manager.remove_session(trigger_sessions[0].session_id)

        tracemalloc.stop()


class TestSessionMemoryPatterns:
    """Test memory usage patterns during session lifecycle."""

    @pytest.mark.asyncio()
    async def test_session_memory_growth_patterns(self):
        """Test memory growth patterns as sessions are modified."""
        session_manager = SessionManager(max_sessions=50, session_ttl_hours=1)

        tracemalloc.start()

        # Create a session for testing
        session = session_manager.create_session(
            "Memory Pattern Test",
            ["MemOpt1", "MemOpt2", "MemOpt3"],
        )

        snapshots = {}

        # Baseline
        gc.collect()
        snapshots["baseline"] = tracemalloc.take_snapshot()

        # Add criteria progressively
        for i in range(10):
            criterion = Criterion(
                name=f"MemoryCriterion{i}",
                description=f"Memory test criterion {i} with longer description to use more memory",
                weight=1.0 + (i * 0.1),
            )
            session.add_criterion(criterion)

            if i in [2, 5, 9]:  # Take snapshots at intervals
                gc.collect()
                snapshots[f"criteria_{i + 1}"] = tracemalloc.take_snapshot()

        # Add evaluation data progressively
        for i in range(5):
            eval_data = {}
            for j in range(min(i + 1, len(session.criteria))):
                criterion_name = f"MemoryCriterion{j}"
                eval_data[criterion_name] = {
                    opt: f"Evaluation data for {opt} on {criterion_name} iteration {i}"
                    for opt in session.options
                }

            session.record_evaluation(eval_data)

            if i in [1, 4]:  # Take snapshots
                gc.collect()
                snapshots[f"evaluations_{i + 1}"] = tracemalloc.take_snapshot()

        # Add scores progressively
        score_count = 0
        for criterion_name in list(session.criteria.keys())[:5]:  # First 5 criteria
            for option_name in session.options:
                score = Score(
                    criterion_name=criterion_name,
                    option_name=option_name,
                    score=7.5 + (score_count * 0.1),
                    justification=f"Memory test score {score_count} with detailed justification text",
                )
                session.options[option_name].add_score(score)
                score_count += 1

        gc.collect()
        snapshots["with_scores"] = tracemalloc.take_snapshot()

        # Calculate memory differences
        baseline = snapshots["baseline"]
        memory_progression = {}

        for snapshot_name, snapshot in snapshots.items():
            if snapshot_name != "baseline":
                diff = snapshot.compare_to(baseline, "lineno")
                memory_growth = sum(stat.size_diff for stat in diff)
                memory_progression[snapshot_name] = memory_growth

        for stage in memory_progression:
            pass

        # Analyze growth pattern
        criteria_stages = [("criteria_3", 3), ("criteria_6", 6), ("criteria_10", 10)]
        eval_stages = [("evaluations_2", 2), ("evaluations_5", 5)]

        # Memory should grow roughly linearly with criteria count
        for i, (stage, count) in enumerate(criteria_stages[1:], 1):
            prev_stage, prev_count = criteria_stages[i - 1]
            current_growth = memory_progression[stage]
            prev_growth = memory_progression[prev_stage]

            growth_per_criterion = (current_growth - prev_growth) / (count - prev_count)

            # Should be reasonable per criterion (less than 10KB each)
            max_memory_per_criterion = 10 * 1024
            assert (
                growth_per_criterion < max_memory_per_criterion
            ), f"Excessive memory per criterion: {growth_per_criterion:.0f} bytes"

        # Memory should grow with evaluations
        if len(eval_stages) > 1:
            eval_growth_diff = (
                memory_progression[eval_stages[1][0]] - memory_progression[eval_stages[0][0]]
            )
            eval_count_diff = eval_stages[1][1] - eval_stages[0][1]
            growth_per_evaluation = eval_growth_diff / eval_count_diff

            # Should be reasonable per evaluation (less than 50KB each)
            max_memory_per_evaluation = 50 * 1024
            assert (
                growth_per_evaluation < max_memory_per_evaluation
            ), f"Excessive memory per evaluation: {growth_per_evaluation:.0f} bytes"

        # Total memory growth should be reasonable
        total_growth = memory_progression["with_scores"]
        max_total_growth = 1024 * 1024  # 1MB max for this test session
        if total_growth > max_total_growth:
            pass

        # Cleanup
        session_manager.remove_session(session.session_id)

        # Verify memory is recovered after cleanup
        gc.collect()
        final_snapshot = tracemalloc.take_snapshot()
        final_diff = final_snapshot.compare_to(baseline, "lineno")
        final_growth = sum(stat.size_diff for stat in final_diff)

        # Most memory should be recovered (allow some overhead)
        max_residual_memory = total_growth * 0.2  # 20% residual is acceptable
        if final_growth > max_residual_memory:
            pass

        tracemalloc.stop()


class TestSessionLookupPerformance:
    """Test session lookup and search performance."""

    @pytest.mark.asyncio()
    async def test_session_lookup_performance_scaling(self):
        """Test session lookup performance as number of sessions grows."""
        max_sessions = 500
        session_manager = SessionManager(max_sessions=max_sessions, session_ttl_hours=1)

        # Test at different session counts
        test_points = [50, 100, 200, 400]
        lookup_results = {}

        sessions_pool = []

        for target_count in test_points:
            # Create sessions up to target count
            while len(sessions_pool) < target_count:
                i = len(sessions_pool)
                session = session_manager.create_session(
                    f"Lookup Test {i}",
                    [f"LookupOpt{i}A", f"LookupOpt{i}B"],
                )
                sessions_pool.append(session)

            # Test lookup performance
            lookup_times = []

            # Test random lookups
            import random

            test_sessions = random.sample(sessions_pool, min(100, len(sessions_pool)))

            for test_session in test_sessions:
                start_time = time.perf_counter()
                retrieved = session_manager.get_session(test_session.session_id)
                end_time = time.perf_counter()

                lookup_times.append(end_time - start_time)
                assert retrieved is not None
                assert retrieved.session_id == test_session.session_id

            # Test listing performance
            start_time = time.perf_counter()
            all_sessions = session_manager.list_active_sessions()
            end_time = time.perf_counter()
            listing_time = end_time - start_time

            # Record results
            lookup_results[target_count] = {
                "mean_lookup_time": mean(lookup_times),
                "median_lookup_time": median(lookup_times),
                "max_lookup_time": max(lookup_times),
                "listing_time": listing_time,
                "lookup_count": len(lookup_times),
            }

            assert len(all_sessions) == target_count

        # Analyze scaling behavior

        for count, results in lookup_results.items():
            results["mean_lookup_time"] * 1000
            results["max_lookup_time"] * 1000
            results["listing_time"] * 1000

        # Performance assertions
        for count, results in lookup_results.items():
            # Lookup should remain fast regardless of session count
            max_lookup_time = 0.005  # 5ms max
            assert (
                results["mean_lookup_time"] < max_lookup_time
            ), f"Lookup too slow at {count} sessions: {results['mean_lookup_time'] * 1000:.2f}ms"

            # Listing time should scale reasonably
            max_listing_time = 0.001 * count / 50  # Scale with session count
            assert (
                results["listing_time"] < max_listing_time
            ), f"Listing too slow at {count} sessions: {results['listing_time'] * 1000:.2f}ms"

        # Cleanup all sessions
        for session in sessions_pool:
            session_manager.remove_session(session.session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output
