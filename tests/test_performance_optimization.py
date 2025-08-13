"""Performance optimization tests for Decision Matrix MCP models.

This module contains comprehensive benchmarks to verify performance improvements
in the optimized decision matrix algorithms. Tests measure both time complexity
and memory usage improvements.

Test Coverage:
    - Matrix generation performance with large datasets
    - Score calculation optimization verification
    - Caching effectiveness measurement
    - Memory usage optimization validation
    - Regression tests for algorithm correctness

Benchmarks:
    - Small dataset: 10 options, 5 criteria
    - Medium dataset: 50 options, 10 criteria  
    - Large dataset: 100 options, 20 criteria
    - XL dataset: 500 options, 50 criteria
"""

import time
import tracemalloc
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from decision_matrix_mcp.models import (
    Criterion,
    DecisionSession,
    ModelBackend,
    Option,
    Score,
)


class TestPerformanceOptimizations:
    """Test suite for performance optimization verification."""

    def setup_method(self):
        """Set up test fixtures for performance testing."""
        self.session = DecisionSession(
            session_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            topic="Performance Testing Decision Analysis"
        )

    def create_test_data(self, num_options: int, num_criteria: int) -> None:
        """Create test data with specified number of options and criteria.
        
        Args:
            num_options: Number of options to create
            num_criteria: Number of criteria to create
        """
        # Add options
        for i in range(num_options):
            self.session.add_option(f"Option_{i}", f"Description for option {i}")
        
        # Add criteria
        for i in range(num_criteria):
            criterion = Criterion(
                name=f"criterion_{i}",
                description=f"Evaluation criterion {i}",
                weight=1.0 + (i * 0.1),  # Varying weights
                model_backend=ModelBackend.BEDROCK
            )
            self.session.add_criterion(criterion)
        
        # Add scores for all option-criterion pairs
        for option_name, option in self.session.options.items():
            for criterion_name in self.session.criteria.keys():
                score = Score(
                    criterion_name=criterion_name,
                    option_name=option_name,
                    score=7.5 + (hash(f"{option_name}_{criterion_name}") % 30) / 10,  # 7.5-10.0 range
                    justification=f"Performance test score for {option_name} on {criterion_name}"
                )
                option.add_score(score)

    def measure_performance(self, func, *args, **kwargs) -> tuple[float, int, any]:
        """Measure execution time and memory usage of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (execution_time_seconds, memory_usage_bytes, result)
        """
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        return execution_time, peak, result

    def test_small_dataset_performance(self):
        """Test performance with small dataset (10 options, 5 criteria)."""
        self.create_test_data(10, 5)
        
        execution_time, memory_usage, result = self.measure_performance(
            self.session.get_decision_matrix
        )
        
        # Performance assertions
        assert execution_time < 0.1, f"Small dataset should complete in <0.1s, took {execution_time:.3f}s"
        assert memory_usage < 1024 * 1024, f"Memory usage should be <1MB, used {memory_usage} bytes"
        assert "error" not in result
        assert len(result["rankings"]) == 10
        assert len(result["criteria_weights"]) == 5

    def test_medium_dataset_performance(self):
        """Test performance with medium dataset (50 options, 10 criteria)."""
        self.create_test_data(50, 10)
        
        execution_time, memory_usage, result = self.measure_performance(
            self.session.get_decision_matrix
        )
        
        # Performance assertions
        assert execution_time < 0.5, f"Medium dataset should complete in <0.5s, took {execution_time:.3f}s"
        assert memory_usage < 5 * 1024 * 1024, f"Memory usage should be <5MB, used {memory_usage} bytes"
        assert "error" not in result
        assert len(result["rankings"]) == 50
        assert len(result["criteria_weights"]) == 10

    def test_large_dataset_performance(self):
        """Test performance with large dataset (100 options, 20 criteria)."""
        self.create_test_data(100, 20)
        
        execution_time, memory_usage, result = self.measure_performance(
            self.session.get_decision_matrix
        )
        
        # Performance assertions - should be significantly faster than O(n²)
        assert execution_time < 1.0, f"Large dataset should complete in <1.0s, took {execution_time:.3f}s"
        assert memory_usage < 10 * 1024 * 1024, f"Memory usage should be <10MB, used {memory_usage} bytes"
        assert "error" not in result
        assert len(result["rankings"]) == 100
        assert len(result["criteria_weights"]) == 20

    def test_caching_effectiveness(self):
        """Test that caching provides significant performance improvements."""
        self.create_test_data(50, 10)
        
        # First call - no cache
        execution_time_1, _, result_1 = self.measure_performance(
            self.session.get_decision_matrix
        )
        
        # Second call - should use cache
        execution_time_2, _, result_2 = self.measure_performance(
            self.session.get_decision_matrix
        )
        
        # Cache should provide significant speedup
        speedup_ratio = execution_time_1 / execution_time_2
        assert speedup_ratio > 2.0, f"Cache should provide >2x speedup, got {speedup_ratio:.2f}x"
        
        # Results should be identical
        assert result_1 == result_2

    def test_cache_invalidation(self):
        """Test that cache is properly invalidated when data changes."""
        self.create_test_data(20, 5)
        
        # Get initial matrix
        matrix_1 = self.session.get_decision_matrix()
        
        # Add new option (should invalidate cache)
        self.session.add_option("New_Option", "A newly added option")
        
        # Add score for new option
        for criterion_name in self.session.criteria.keys():
            score = Score(
                criterion_name=criterion_name,
                option_name="New_Option",
                score=9.0,
                justification="High score for new option"
            )
            self.session.options["New_Option"].add_score(score)
        
        # Get matrix again - should be different
        matrix_2 = self.session.get_decision_matrix()
        
        assert len(matrix_2["rankings"]) == len(matrix_1["rankings"]) + 1
        assert matrix_2["rankings"][0]["option"] == "New_Option"  # Should be winner with 9.0 scores

    def test_weighted_total_optimization(self):
        """Test optimized weighted total calculation performance."""
        self.create_test_data(100, 20)
        
        option = list(self.session.options.values())[0]
        
        # Measure weighted total calculation performance
        execution_time, memory_usage, result = self.measure_performance(
            option.get_weighted_total, self.session.criteria
        )
        
        assert execution_time < 0.01, f"Weighted total should complete in <0.01s, took {execution_time:.4f}s"
        assert isinstance(result, float)
        assert 0.0 <= result <= 10.0

    def test_score_breakdown_optimization(self):
        """Test optimized score breakdown generation performance."""
        self.create_test_data(50, 15)
        
        option = list(self.session.options.values())[0]
        
        # Measure score breakdown performance
        execution_time, memory_usage, result = self.measure_performance(
            option.get_score_breakdown, self.session.criteria
        )
        
        assert execution_time < 0.01, f"Score breakdown should complete in <0.01s, took {execution_time:.4f}s"
        assert len(result) == 15  # Should have breakdown for all criteria
        assert all("criterion" in item for item in result)
        assert all("weighted_score" in item for item in result)

    def test_memory_efficiency_slots(self):
        """Test that __slots__ optimization reduces memory usage."""
        # Create many Score objects to test memory efficiency
        scores = []
        
        tracemalloc.start()
        
        for i in range(1000):
            score = Score(
                criterion_name=f"criterion_{i % 10}",
                option_name=f"option_{i % 20}",
                score=float(i % 10),
                justification=f"Test score {i}"
            )
            scores.append(score)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # With __slots__, memory usage should be significantly reduced
        bytes_per_score = peak / len(scores)
        assert bytes_per_score < 1000, f"Each Score should use <1KB, using {bytes_per_score:.0f} bytes"

    def test_algorithm_correctness(self):
        """Verify that optimizations don't affect algorithm correctness."""
        self.create_test_data(20, 8)
        
        # Get decision matrix
        matrix = self.session.get_decision_matrix()
        
        # Verify structure
        assert "matrix" in matrix
        assert "rankings" in matrix
        assert "recommendation" in matrix
        assert "criteria_weights" in matrix
        
        # Verify matrix dimensions
        assert len(matrix["matrix"]) == 20  # All options
        for option_scores in matrix["matrix"].values():
            assert len(option_scores) == 8  # All criteria
        
        # Verify rankings are sorted correctly
        rankings = matrix["rankings"]
        for i in range(len(rankings) - 1):
            assert rankings[i]["weighted_total"] >= rankings[i + 1]["weighted_total"]
        
        # Verify weights are correct
        weights = matrix["criteria_weights"]
        for i, (name, weight) in enumerate(weights.items()):
            expected_weight = 1.0 + (i * 0.1)
            assert abs(weight - expected_weight) < 0.001

    def test_concurrent_access_safety(self):
        """Test that caching is safe for concurrent access."""
        import threading
        import time
        
        self.create_test_data(30, 6)
        
        results = []
        errors = []
        
        def worker():
            try:
                matrix = self.session.get_decision_matrix()
                results.append(matrix)
            except Exception as e:
                errors.append(e)
        
        # Launch multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"
        assert len(results) == 5
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    @pytest.mark.benchmark
    def test_xl_dataset_benchmark(self):
        """Benchmark test with extra large dataset (500 options, 50 criteria)."""
        self.create_test_data(500, 50)
        
        execution_time, memory_usage, result = self.measure_performance(
            self.session.get_decision_matrix
        )
        
        # Even with XL dataset, should complete reasonably fast
        assert execution_time < 5.0, f"XL dataset should complete in <5.0s, took {execution_time:.3f}s"
        assert memory_usage < 50 * 1024 * 1024, f"Memory usage should be <50MB, used {memory_usage} bytes"
        assert "error" not in result
        assert len(result["rankings"]) == 500
        assert len(result["criteria_weights"]) == 50
        
        print(f"\nXL Dataset Performance:")
        print(f"  Execution time: {execution_time:.3f} seconds")
        print(f"  Memory usage: {memory_usage / (1024*1024):.1f} MB")
        print(f"  Options processed: 500")
        print(f"  Criteria processed: 50")
        print(f"  Total evaluations: 25,000")

    def test_cached_property_performance(self):
        """Test that cached property provides additional performance benefits."""
        self.create_test_data(40, 8)
        
        # First access via cached property
        start_time = time.perf_counter()
        matrix_1 = self.session.decision_matrix
        time_1 = time.perf_counter() - start_time
        
        # Second access should be nearly instant
        start_time = time.perf_counter()
        matrix_2 = self.session.decision_matrix
        time_2 = time.perf_counter() - start_time
        
        # Cached property should be much faster
        assert time_2 < time_1 / 10, f"Cached property should be >10x faster, got {time_1/time_2:.1f}x"
        assert matrix_1 == matrix_2