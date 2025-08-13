"""
Performance tests for LLM backend operations.

These tests measure:
- Response time measurement for each backend
- Throughput testing under load
- Retry logic performance
- Error handling performance impact
- Backend comparison metrics
"""

import asyncio
from statistics import mean, median, stdev
import time

import pytest

from decision_matrix_mcp.backends.bedrock import BedrockBackend
from decision_matrix_mcp.backends.litellm import LiteLLMBackend
from decision_matrix_mcp.backends.ollama import OllamaBackend
from decision_matrix_mcp.exceptions import LLMBackendError


class BackendPerformanceProfiler:
    """Profiler for backend performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.response_times = {}
        self.error_counts = {}
        self.retry_counts = {}
        self.throughput_metrics = {}

    def record_response_time(self, backend_name: str, operation: str, duration: float):
        """Record response time for a backend operation."""
        key = f"{backend_name}_{operation}"
        if key not in self.response_times:
            self.response_times[key] = []
        self.response_times[key].append(duration)

    def record_error(self, backend_name: str, error_type: str):
        """Record an error for a backend."""
        key = f"{backend_name}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    def record_retry(self, backend_name: str, attempt_count: int):
        """Record retry attempt for a backend."""
        if backend_name not in self.retry_counts:
            self.retry_counts[backend_name] = []
        self.retry_counts[backend_name].append(attempt_count)

    def get_stats(self, backend_name: str, operation: str) -> dict:
        """Get statistics for a backend operation."""
        key = f"{backend_name}_{operation}"
        if key not in self.response_times or not self.response_times[key]:
            return {}

        times = self.response_times[key]
        return {
            "count": len(times),
            "mean": mean(times),
            "median": median(times),
            "min": min(times),
            "max": max(times),
            "std": stdev(times) if len(times) > 1 else 0.0,
            "p95": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0],
            "p99": sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else times[0],
        }

    def print_backend_comparison(self, backends: list[str]):
        """Print performance comparison across backends."""
        print("\n" + "=" * 80)
        print("BACKEND PERFORMANCE COMPARISON")
        print("=" * 80)

        print(
            f"\n{'Backend':<15} {'Count':<8} {'Mean':<10} {'Median':<10} {'P95':<10} {'P99':<10} {'Std':<10}",
        )
        print("-" * 80)

        for backend in backends:
            stats = self.get_stats(backend, "generate")
            if stats:
                print(
                    f"{backend:<15} {stats['count']:<8} {stats['mean']*1000:<10.1f} "
                    f"{stats['median']*1000:<10.1f} {stats['p95']*1000:<10.1f} "
                    f"{stats['p99']*1000:<10.1f} {stats['std']*1000:<10.1f}",
                )


class MockBackendFactory:
    """Factory for creating mock backends with configurable performance characteristics."""

    @staticmethod
    def create_mock_bedrock(
        base_latency: float = 0.1,
        latency_variance: float = 0.02,
        error_rate: float = 0.0,
    ) -> BedrockBackend:
        """Create mock Bedrock backend with specified performance characteristics."""
        backend = BedrockBackend()

        async def mock_generate(*args, **kwargs):
            # Simulate latency
            latency = base_latency + (hash(str(args)) % 100) / 1000 * latency_variance
            await asyncio.sleep(latency)

            # Simulate errors
            if error_rate > 0 and (hash(str(args)) % 100) / 100 < error_rate:
                raise LLMBackendError("Mock Bedrock error", "Simulated error")

            return "8.5: Mock Bedrock response with detailed evaluation"

        backend.generate = mock_generate
        return backend

    @staticmethod
    def create_mock_litellm(
        base_latency: float = 0.05,
        latency_variance: float = 0.01,
        error_rate: float = 0.0,
    ) -> LiteLLMBackend:
        """Create mock LiteLLM backend with specified performance characteristics."""
        backend = LiteLLMBackend()

        async def mock_generate(*args, **kwargs):
            # Simulate latency
            latency = base_latency + (hash(str(args)) % 100) / 1000 * latency_variance
            await asyncio.sleep(latency)

            # Simulate errors
            if error_rate > 0 and (hash(str(args)) % 100) / 100 < error_rate:
                raise LLMBackendError("Mock LiteLLM error", "Simulated error")

            return "7.0: Mock LiteLLM response with fast evaluation"

        backend.generate = mock_generate
        return backend

    @staticmethod
    def create_mock_ollama(
        base_latency: float = 0.2,
        latency_variance: float = 0.05,
        error_rate: float = 0.0,
    ) -> OllamaBackend:
        """Create mock Ollama backend with specified performance characteristics."""
        backend = OllamaBackend()

        async def mock_generate(*args, **kwargs):
            # Simulate latency (typically slower for local models)
            latency = base_latency + (hash(str(args)) % 100) / 1000 * latency_variance
            await asyncio.sleep(latency)

            # Simulate errors
            if error_rate > 0 and (hash(str(args)) % 100) / 100 < error_rate:
                raise LLMBackendError("Mock Ollama error", "Simulated error")

            return "6.5: Mock Ollama response with local model evaluation"

        backend.generate = mock_generate
        return backend


class TestBackendResponseTimes:
    """Test response time characteristics of different backends."""

    @pytest.fixture()
    def profiler(self):
        """Create performance profiler."""
        return BackendPerformanceProfiler()

    @pytest.mark.parametrize(
        "backend_type,expected_max_latency",
        [
            ("bedrock", 0.15),  # Bedrock should respond within 150ms (mock)
            ("litellm", 0.08),  # LiteLLM should be faster (mock)
            ("ollama", 0.30),  # Ollama can be slower (local model, mock)
        ],
    )
    @pytest.mark.asyncio()
    async def test_backend_response_time_characteristics(
        self,
        profiler,
        backend_type,
        expected_max_latency,
    ):
        """Test response time characteristics for each backend type."""
        # Create mock backend with realistic performance characteristics
        if backend_type == "bedrock":
            backend = MockBackendFactory.create_mock_bedrock()
        elif backend_type == "litellm":
            backend = MockBackendFactory.create_mock_litellm()
        elif backend_type == "ollama":
            backend = MockBackendFactory.create_mock_ollama()
        else:
            pytest.skip(f"Unknown backend type: {backend_type}")

        # Test multiple requests to get statistical data
        num_requests = 20

        for i in range(num_requests):
            start_time = time.perf_counter()

            try:
                if backend_type == "litellm":
                    response = await backend.generate(
                        system_prompt=f"Evaluate option {i}",
                        user_prompt=f"Rate this option for criterion {i}",
                        model="gpt-3.5-turbo",
                    )
                elif backend_type == "ollama":
                    response = await backend.generate(
                        system_prompt=f"Evaluate option {i}",
                        user_prompt=f"Rate this option for criterion {i}",
                        model="llama3.2:3b",
                    )
                else:  # bedrock
                    response = await backend.generate(
                        system_prompt=f"Evaluate option {i}",
                        user_prompt=f"Rate this option for criterion {i}",
                    )

                end_time = time.perf_counter()
                duration = end_time - start_time

                profiler.record_response_time(backend_type, "generate", duration)

                # Verify response is valid
                assert isinstance(response, str)
                assert len(response) > 0

            except LLMBackendError:
                profiler.record_error(backend_type, "llm_error")
                # Continue with other requests

        # Analyze response time statistics
        stats = profiler.get_stats(backend_type, "generate")

        print(f"\n{backend_type.upper()} Response Time Statistics ({num_requests} requests):")
        if stats:
            print(f"  Mean: {stats['mean']*1000:.1f}ms")
            print(f"  Median: {stats['median']*1000:.1f}ms")
            print(f"  Min: {stats['min']*1000:.1f}ms")
            print(f"  Max: {stats['max']*1000:.1f}ms")
            print(f"  P95: {stats['p95']*1000:.1f}ms")
            print(f"  P99: {stats['p99']*1000:.1f}ms")
            print(f"  Std Dev: {stats['std']*1000:.1f}ms")

            # Performance assertions
            assert (
                stats["mean"] < expected_max_latency
            ), f"{backend_type} mean response time too high: {stats['mean']*1000:.1f}ms"

            assert (
                stats["p95"] < expected_max_latency * 1.5
            ), f"{backend_type} P95 response time too high: {stats['p95']*1000:.1f}ms"

            # Response times should be reasonably consistent
            max_std_dev = expected_max_latency * 0.3
            assert (
                stats["std"] < max_std_dev
            ), f"{backend_type} response time too variable: {stats['std']*1000:.1f}ms std dev"
        else:
            pytest.fail(f"No successful requests recorded for {backend_type}")

    @pytest.mark.asyncio()
    async def test_backend_performance_comparison(self, profiler):
        """Test and compare performance across all backend types."""
        backends = {
            "bedrock": MockBackendFactory.create_mock_bedrock(base_latency=0.08),
            "litellm": MockBackendFactory.create_mock_litellm(base_latency=0.04),
            "ollama": MockBackendFactory.create_mock_ollama(base_latency=0.15),
        }

        num_requests = 15

        # Test each backend
        for backend_name, backend in backends.items():
            for i in range(num_requests):
                start_time = time.perf_counter()

                try:
                    if backend_name == "litellm":
                        response = await backend.generate(
                            "System prompt",
                            f"User prompt {i}",
                            model="gpt-3.5-turbo",
                        )
                    elif backend_name == "ollama":
                        response = await backend.generate(
                            "System prompt",
                            f"User prompt {i}",
                            model="llama3.2:3b",
                        )
                    else:  # bedrock
                        response = await backend.generate("System prompt", f"User prompt {i}")

                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    profiler.record_response_time(backend_name, "generate", duration)

                except LLMBackendError:
                    profiler.record_error(backend_name, "error")

        # Print comparison
        profiler.print_backend_comparison(["bedrock", "litellm", "ollama"])

        # Verify relative performance expectations
        bedrock_stats = profiler.get_stats("bedrock", "generate")
        litellm_stats = profiler.get_stats("litellm", "generate")
        ollama_stats = profiler.get_stats("ollama", "generate")

        # LiteLLM should generally be fastest in this mock setup
        if litellm_stats and bedrock_stats:
            assert (
                litellm_stats["mean"] <= bedrock_stats["mean"] * 1.2
            ), "LiteLLM should be competitive with Bedrock"

        # Ollama might be slower but should still be reasonable
        if ollama_stats:
            assert ollama_stats["mean"] < 0.25, "Ollama response time should be reasonable"


class TestBackendThroughput:
    """Test backend throughput under concurrent load."""

    @pytest.mark.asyncio()
    async def test_concurrent_request_throughput(self):
        """Test throughput of concurrent requests to backends."""
        profiler = BackendPerformanceProfiler()

        # Create backends with different characteristics
        backends = {
            "bedrock": MockBackendFactory.create_mock_bedrock(
                base_latency=0.05,
                latency_variance=0.01,
            ),
            "litellm": MockBackendFactory.create_mock_litellm(
                base_latency=0.03,
                latency_variance=0.005,
            ),
            "ollama": MockBackendFactory.create_mock_ollama(
                base_latency=0.08,
                latency_variance=0.02,
            ),
        }

        concurrent_requests = 10

        for backend_name, backend in backends.items():
            print(
                f"\nTesting {backend_name} throughput with {concurrent_requests} concurrent requests...",
            )

            async def make_request(request_id):
                """Make a single request."""
                start_time = time.perf_counter()

                try:
                    if backend_name == "litellm":
                        response = await backend.generate(
                            f"System {request_id}",
                            f"User {request_id}",
                            model="gpt-3.5-turbo",
                        )
                    elif backend_name == "ollama":
                        response = await backend.generate(
                            f"System {request_id}",
                            f"User {request_id}",
                            model="llama3.2:3b",
                        )
                    else:  # bedrock
                        response = await backend.generate(
                            f"System {request_id}",
                            f"User {request_id}",
                        )

                    end_time = time.perf_counter()
                    return True, end_time - start_time, len(response)

                except LLMBackendError as e:
                    end_time = time.perf_counter()
                    return False, end_time - start_time, str(e)

            # Measure concurrent throughput
            start_time = time.perf_counter()

            tasks = [make_request(i) for i in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_duration = end_time - start_time

            # Analyze results
            successful_requests = [r for r in results if r[0]]
            failed_requests = [r for r in results if not r[0]]

            if successful_requests:
                response_times = [r[1] for r in successful_requests]
                mean_response_time = mean(response_times)
                throughput = len(successful_requests) / total_duration

                print(f"  Successful requests: {len(successful_requests)}/{concurrent_requests}")
                print(f"  Total time: {total_duration*1000:.1f}ms")
                print(f"  Mean response time: {mean_response_time*1000:.1f}ms")
                print(f"  Throughput: {throughput:.1f} requests/sec")

                # Record metrics
                for response_time in response_times:
                    profiler.record_response_time(backend_name, "concurrent", response_time)

                profiler.throughput_metrics[backend_name] = {
                    "throughput": throughput,
                    "total_duration": total_duration,
                    "successful_requests": len(successful_requests),
                    "failed_requests": len(failed_requests),
                }

                # Performance assertions
                min_throughput = 5  # At least 5 requests/sec for concurrent load
                assert (
                    throughput > min_throughput
                ), f"{backend_name} throughput too low: {throughput:.1f} < {min_throughput}"

                # Most requests should succeed
                success_rate = len(successful_requests) / concurrent_requests
                min_success_rate = 0.8  # 80% success rate minimum
                assert (
                    success_rate >= min_success_rate
                ), f"{backend_name} success rate too low: {success_rate:.2f} < {min_success_rate}"

            else:
                pytest.fail(f"No successful requests for {backend_name}")

        # Print throughput comparison
        print(f"\nThroughput Comparison ({concurrent_requests} concurrent requests):")
        print(f"{'Backend':<15} {'Throughput':<12} {'Success Rate':<12} {'Mean Time':<12}")
        print("-" * 55)

        for backend_name in backends:
            if backend_name in profiler.throughput_metrics:
                metrics = profiler.throughput_metrics[backend_name]
                success_rate = metrics["successful_requests"] / concurrent_requests
                stats = profiler.get_stats(backend_name, "concurrent")
                mean_time = stats["mean"] * 1000 if stats else 0

                print(
                    f"{backend_name:<15} {metrics['throughput']:<12.1f} "
                    f"{success_rate:<12.2f} {mean_time:<12.1f}",
                )

    @pytest.mark.asyncio()
    async def test_sustained_load_performance(self):
        """Test backend performance under sustained load."""
        profiler = BackendPerformanceProfiler()

        # Test with LiteLLM backend (typically fastest)
        backend = MockBackendFactory.create_mock_litellm(base_latency=0.02, latency_variance=0.005)

        # Sustained load parameters
        duration_seconds = 5
        requests_per_second = 8
        total_requests = duration_seconds * requests_per_second

        print(f"\nSustained load test: {total_requests} requests over {duration_seconds}s...")

        async def sustained_request_batch():
            """Make requests at specified rate."""
            request_count = 0
            start_time = time.perf_counter()

            while request_count < total_requests:
                batch_start = time.perf_counter()

                # Send batch of requests
                batch_size = min(requests_per_second, total_requests - request_count)
                batch_tasks = []

                for i in range(batch_size):

                    async def make_sustained_request(req_id=request_count + i):
                        req_start = time.perf_counter()
                        try:
                            response = await backend.generate(
                                f"Sustained system {req_id}",
                                f"Sustained user {req_id}",
                                model="gpt-3.5-turbo",
                            )
                            req_end = time.perf_counter()
                            profiler.record_response_time(
                                "sustained",
                                "generate",
                                req_end - req_start,
                            )
                            return True, req_end - req_start
                        except LLMBackendError:
                            req_end = time.perf_counter()
                            profiler.record_error("sustained", "error")
                            return False, req_end - req_start

                    batch_tasks.append(make_sustained_request())

                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks)
                request_count += batch_size

                # Rate limiting - wait to maintain target rate
                batch_end = time.perf_counter()
                batch_duration = batch_end - batch_start
                target_batch_duration = 1.0  # 1 second per batch

                if batch_duration < target_batch_duration:
                    await asyncio.sleep(target_batch_duration - batch_duration)

            end_time = time.perf_counter()
            return end_time - start_time, request_count

        # Run sustained load test
        total_duration, completed_requests = await sustained_request_batch()

        # Analyze sustained performance
        stats = profiler.get_stats("sustained", "generate")

        if stats and stats["count"] > 0:
            actual_throughput = completed_requests / total_duration

            print(f"  Completed requests: {completed_requests}/{total_requests}")
            print(f"  Total duration: {total_duration:.2f}s")
            print(f"  Target throughput: {requests_per_second:.1f} req/s")
            print(f"  Actual throughput: {actual_throughput:.1f} req/s")
            print(f"  Mean response time: {stats['mean']*1000:.1f}ms")
            print(f"  P95 response time: {stats['p95']*1000:.1f}ms")
            print(f"  Response time std: {stats['std']*1000:.1f}ms")

            # Performance assertions for sustained load
            min_completion_rate = 0.9  # Complete at least 90% of requests
            completion_rate = completed_requests / total_requests
            assert (
                completion_rate >= min_completion_rate
            ), f"Low completion rate under sustained load: {completion_rate:.2f}"

            # Response times should remain stable under sustained load
            max_mean_response_time = 0.05  # 50ms max mean response time
            assert (
                stats["mean"] < max_mean_response_time
            ), f"Response time degraded under sustained load: {stats['mean']*1000:.1f}ms"

            # P95 shouldn't be too much higher than mean
            max_p95_ratio = 3.0  # P95 should be less than 3x mean
            p95_ratio = stats["p95"] / stats["mean"] if stats["mean"] > 0 else float("inf")
            assert (
                p95_ratio < max_p95_ratio
            ), f"High P95/mean ratio under sustained load: {p95_ratio:.2f}"

        else:
            pytest.fail("No successful requests during sustained load test")


class TestBackendErrorHandling:
    """Test backend error handling and recovery performance."""

    @pytest.mark.asyncio()
    async def test_error_handling_performance(self):
        """Test performance impact of error handling and retries."""
        profiler = BackendPerformanceProfiler()

        # Create backends with different error rates
        error_rates = [0.0, 0.2, 0.5]  # 0%, 20%, 50% error rates

        for error_rate in error_rates:
            backend = MockBackendFactory.create_mock_bedrock(
                base_latency=0.05,
                error_rate=error_rate,
            )

            num_requests = 20
            successful_responses = []
            error_responses = []

            print(f"\nTesting error handling with {error_rate*100:.0f}% error rate...")

            for i in range(num_requests):
                start_time = time.perf_counter()

                try:
                    response = await backend.generate(
                        f"Error test system {i}",
                        f"Error test user {i}",
                    )
                    end_time = time.perf_counter()

                    successful_responses.append(end_time - start_time)
                    profiler.record_response_time(
                        f"error_{error_rate}",
                        "success",
                        end_time - start_time,
                    )

                except LLMBackendError:
                    end_time = time.perf_counter()

                    error_responses.append(end_time - start_time)
                    profiler.record_response_time(
                        f"error_{error_rate}",
                        "error",
                        end_time - start_time,
                    )
                    profiler.record_error(f"error_{error_rate}", "llm_error")

            # Analyze error handling performance
            success_count = len(successful_responses)
            error_count = len(error_responses)
            actual_error_rate = error_count / num_requests

            print(f"  Successful requests: {success_count}/{num_requests}")
            print(f"  Error requests: {error_count}/{num_requests}")
            print(f"  Actual error rate: {actual_error_rate:.2f}")

            if successful_responses:
                mean_success_time = mean(successful_responses)
                print(f"  Mean success time: {mean_success_time*1000:.1f}ms")

            if error_responses:
                mean_error_time = mean(error_responses)
                print(f"  Mean error time: {mean_error_time*1000:.1f}ms")

                # Error handling should be reasonably fast
                max_error_time = 0.1  # 100ms max for error handling
                assert (
                    mean_error_time < max_error_time
                ), f"Error handling too slow: {mean_error_time*1000:.1f}ms"

            # Verify error rate is as expected (within tolerance)
            error_rate_tolerance = 0.15
            assert (
                abs(actual_error_rate - error_rate) < error_rate_tolerance
            ), f"Error rate mismatch: {actual_error_rate:.2f} vs expected {error_rate:.2f}"

    @pytest.mark.asyncio()
    async def test_retry_logic_performance(self):
        """Test performance of retry logic for failed requests."""
        # This test would be more meaningful with a real backend that implements retry logic
        # For now, we'll test the concept with a mock that simulates retries

        class MockRetryBackend:
            def __init__(self, fail_rate=0.3, max_retries=3):
                self.fail_rate = fail_rate
                self.max_retries = max_retries
                self.attempt_count = 0

            async def generate(self, system_prompt, user_prompt, **kwargs):
                """Mock generate with retry simulation."""
                for attempt in range(self.max_retries + 1):
                    self.attempt_count += 1

                    # Simulate network delay
                    await asyncio.sleep(0.02)

                    # Simulate failure based on fail rate
                    if (hash(f"{system_prompt}{attempt}") % 100) / 100 < self.fail_rate:
                        if attempt < self.max_retries:
                            continue  # Retry
                        raise LLMBackendError("Max retries exceeded", "Simulated failure")
                    return f"Success on attempt {attempt + 1}"

                raise LLMBackendError("Unexpected retry logic error", "Should not reach here")

        profiler = BackendPerformanceProfiler()
        backend = MockRetryBackend(fail_rate=0.4, max_retries=2)

        num_requests = 15

        for i in range(num_requests):
            start_time = time.perf_counter()
            initial_attempt_count = backend.attempt_count

            try:
                response = await backend.generate(f"Retry test {i}", f"User prompt {i}")
                end_time = time.perf_counter()

                attempts_used = backend.attempt_count - initial_attempt_count
                profiler.record_retry("retry_backend", attempts_used)
                profiler.record_response_time("retry_backend", "success", end_time - start_time)

            except LLMBackendError:
                end_time = time.perf_counter()

                attempts_used = backend.attempt_count - initial_attempt_count
                profiler.record_retry("retry_backend", attempts_used)
                profiler.record_response_time("retry_backend", "failure", end_time - start_time)
                profiler.record_error("retry_backend", "max_retries")

        # Analyze retry performance
        retry_stats = profiler.retry_counts.get("retry_backend", [])
        success_stats = profiler.get_stats("retry_backend", "success")
        failure_stats = profiler.get_stats("retry_backend", "failure")

        print("\nRetry Logic Performance:")
        print(f"  Total requests: {num_requests}")
        print(f"  Retry attempts: {retry_stats}")
        print(f"  Mean attempts per request: {mean(retry_stats):.1f}")
        print(f"  Max attempts: {max(retry_stats) if retry_stats else 0}")

        if success_stats:
            print(f"  Successful requests: {success_stats['count']}")
            print(f"  Mean success time: {success_stats['mean']*1000:.1f}ms")

        if failure_stats:
            print(f"  Failed requests: {failure_stats['count']}")
            print(f"  Mean failure time: {failure_stats['mean']*1000:.1f}ms")

        # Retry logic should be reasonably efficient
        if retry_stats:
            mean_attempts = mean(retry_stats)
            assert mean_attempts <= 2.5, f"Too many retry attempts: {mean_attempts:.1f}"

        # Total time including retries should still be reasonable
        if success_stats:
            max_success_time_with_retries = 0.15  # 150ms including retries
            assert (
                success_stats["mean"] < max_success_time_with_retries
            ), f"Success time with retries too high: {success_stats['mean']*1000:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output
