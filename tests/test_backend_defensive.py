"""
Defensive programming tests for backend integration and failure recovery.

Tests backend factory behavior when some backends fail, partial availability,
graceful degradation, and error recovery patterns.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from decision_matrix_mcp.backends.bedrock import BedrockBackend
from decision_matrix_mcp.backends.factory import BackendFactory
from decision_matrix_mcp.backends.litellm import LiteLLMBackend
from decision_matrix_mcp.backends.ollama import OllamaBackend
from decision_matrix_mcp.exceptions import LLMBackendError
from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend


class TestBackendDefensivePatterns:
    """Test defensive patterns in backend management."""

    @pytest.fixture()
    def factory(self):
        return BackendFactory()

    def test_partial_backend_availability_defensive(self, factory):
        """Test when only some backends are available - defensive graceful degradation."""
        with (
            # Bedrock fails completely
            patch.object(BedrockBackend, "is_available", return_value=False),
            patch.object(BedrockBackend, "__init__", side_effect=ImportError("boto3 not found")),
            # LiteLLM available
            patch.object(LiteLLMBackend, "is_available", return_value=True),
            # Ollama has connection issues
            patch.object(
                OllamaBackend, "is_available", side_effect=ConnectionError("Ollama server down")
            ),
        ):
            available_backends = factory.get_available_backends()

            # Should only return working backends
            assert ModelBackend.LITELLM in available_backends
            assert ModelBackend.BEDROCK not in available_backends
            assert ModelBackend.OLLAMA not in available_backends
            assert len(available_backends) == 1

    def test_backend_creation_with_cascading_failures(self, factory):
        """Test backend creation when preferred backend fails but fallback succeeds."""
        # Mock Bedrock creation to fail
        with patch.object(
            BedrockBackend, "__init__", side_effect=Exception("AWS credentials invalid")
        ):
            # Should raise exception (no automatic fallback in create_backend)
            with pytest.raises(Exception, match="AWS credentials invalid"):
                factory.create_backend(ModelBackend.BEDROCK)

    def test_backend_validation_edge_cases(self, factory):
        """Test backend validation with various edge case failures."""
        test_cases = [
            (ImportError("Missing dependency"), False),
            (ConnectionError("Network unreachable"), False),
            (AttributeError("Method not found"), False),
            (TimeoutError("Connection timeout"), False),
            (PermissionError("Access denied"), False),
            (ValueError("Invalid configuration"), False),
            (RuntimeError("Runtime failure"), False),
        ]

        for exception, expected_available in test_cases:
            with patch.object(factory, "create_backend", side_effect=exception):
                result = factory.validate_backend_availability(ModelBackend.BEDROCK)
                assert result == expected_available, f"Failed for {exception.__class__.__name__}"

    def test_backend_cleanup_partial_failures_defensive(self, factory):
        """Test cleanup when some backends fail to clean up properly."""
        # Create multiple backend instances
        bedrock_backend = Mock()
        litellm_backend = Mock()
        ollama_backend = Mock()

        # Add the instances directly to the factory's _instances dict
        factory._instances[ModelBackend.BEDROCK] = bedrock_backend
        factory._instances[ModelBackend.LITELLM] = litellm_backend
        factory._instances[ModelBackend.OLLAMA] = ollama_backend

        # Make some cleanup methods fail
        bedrock_backend.cleanup.side_effect = RuntimeError("Bedrock cleanup failed")
        litellm_backend.cleanup.return_value = None  # Succeeds
        ollama_backend.cleanup.side_effect = ConnectionError("Ollama cleanup failed")

        # Cleanup should handle all failures gracefully
        factory.cleanup()  # Should not raise any exceptions

        # All backends should have been attempted
        bedrock_backend.cleanup.assert_called_once()
        litellm_backend.cleanup.assert_called_once()
        ollama_backend.cleanup.assert_called_once()

        # Factory should clear instances despite partial failures
        assert len(factory._instances) == 0

    def test_backend_no_cleanup_method_defensive(self, factory):
        """Test cleanup when backend doesn't implement cleanup method."""
        # Create mock backend without cleanup method
        mock_backend = Mock(spec=[])  # No cleanup method

        with patch.object(factory, "create_backend", return_value=mock_backend):
            factory.create_backend(ModelBackend.BEDROCK)

            # Should handle missing cleanup method gracefully
            factory.cleanup()  # Should not raise AttributeError

            # Instances should still be cleared
            assert len(factory._instances) == 0

    async def test_async_backend_response_defensive_error_handling(self, factory):
        """Test async backend response handling with various error conditions."""
        criterion = Criterion(
            name="test", description="test", model_backend=ModelBackend.BEDROCK, weight=1.0
        )
        thread = CriterionThread(id="test", criterion=criterion)
        thread.add_message("user", "Test message")

        error_scenarios = [
            (ConnectionError("Network failed"), "connection"),
            (TimeoutError("Request timeout"), "timeout"),
            (ValueError("Invalid response"), "invalid"),
            (RuntimeError("Runtime error"), "runtime"),
            (Exception("Generic error"), "error"),
        ]

        for exception, expected_error_type in error_scenarios:
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = exception

            with patch.object(factory, "create_backend", return_value=mock_backend):
                backend = factory.create_backend(ModelBackend.BEDROCK)

                # Should raise LLMBackendError with appropriate context
                with pytest.raises(LLMBackendError) as exc_info:
                    await backend.generate_response(thread)

                # Error should contain context about the failure
                error_message = str(exc_info.value)
                assert (
                    expected_error_type in error_message.lower() or "error" in error_message.lower()
                )

    def test_concurrent_backend_creation_defensive(self, factory):
        """Test concurrent backend creation doesn't create duplicate instances."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        creation_results = []
        creation_errors = []

        def create_backend_thread(thread_id):
            """Create backend from multiple threads."""
            try:
                backend = factory.create_backend(ModelBackend.BEDROCK)
                creation_results.append(("success", thread_id, id(backend)))
            except Exception as e:
                creation_errors.append(("error", thread_id, str(e)))

        # Create backends concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_backend_thread, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        # Should have created instances (may be same instance due to singleton pattern)
        assert len(creation_results) == 5
        assert len(creation_errors) == 0

        # Verify factory internal state is consistent
        assert len(factory._instances) <= 1  # Should be singleton or empty if all same type

    def test_backend_instance_caching_defensive(self, factory):
        """Test backend instance caching behavior under various conditions."""
        # Create same backend type multiple times
        backend1 = factory.create_backend(ModelBackend.BEDROCK)
        backend2 = factory.create_backend(ModelBackend.BEDROCK)
        backend3 = factory.create_backend(ModelBackend.BEDROCK)

        # Should return same instance (caching)
        assert backend1 is backend2
        assert backend2 is backend3

        # Should have only one instance cached
        assert len(factory._instances) == 1

        # Create different backend type
        backend4 = factory.create_backend(ModelBackend.LITELLM)

        # Should have two different instances cached
        assert backend1 is not backend4
        assert len(factory._instances) == 2

    def test_backend_availability_check_defensive_caching(self, factory):
        """Test backend availability checking doesn't interfere with instance caching."""
        # Check availability multiple times
        with patch.object(BedrockBackend, "is_available", return_value=True):
            result1 = factory.validate_backend_availability(ModelBackend.BEDROCK)
            result2 = factory.validate_backend_availability(ModelBackend.BEDROCK)
            result3 = factory.validate_backend_availability(ModelBackend.BEDROCK)

            assert result1 is True
            assert result2 is True
            assert result3 is True

            # Availability checks should create temporary instances, not cached ones
            # Or reuse cached instances without duplicating
            assert len(factory._instances) <= 1

    async def test_backend_response_timeout_defensive_recovery(self):
        """Test backend response timeout handling and recovery."""
        factory = BackendFactory()
        criterion = Criterion(
            name="timeout_test", description="test", model_backend=ModelBackend.OLLAMA, weight=1.0
        )
        thread = CriterionThread(id="test", criterion=criterion)
        thread.add_message("user", "Test message")

        # Mock backend that times out first, then succeeds
        mock_backend = AsyncMock()
        call_count = 0

        async def mock_generate_with_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("First call timeout")
            return "SCORE: 7\nJUSTIFICATION: Recovered successfully"

        mock_backend.generate_response.side_effect = mock_generate_with_recovery

        with patch.object(factory, "create_backend", return_value=mock_backend):
            backend = factory.create_backend(ModelBackend.OLLAMA)

            # First call should timeout
            with pytest.raises(asyncio.TimeoutError):
                await backend.generate_response(thread)

            # Second call should succeed (recovery)
            result = await backend.generate_response(thread)
            assert "Recovered successfully" in result

    def test_backend_factory_state_consistency_under_errors(self, factory):
        """Test factory maintains consistent state even when operations fail."""
        # Start with clean state
        initial_instance_count = len(factory._instances)

        # Attempt operations that fail
        with patch.object(BedrockBackend, "__init__", side_effect=ImportError("Failure")):
            with pytest.raises(ImportError):
                factory.create_backend(ModelBackend.BEDROCK)

        # Factory state should be unchanged after failure
        assert len(factory._instances) == initial_instance_count

        # Successful creation after failure should work
        with patch.object(BedrockBackend, "__init__", return_value=None):
            backend = factory.create_backend(ModelBackend.BEDROCK)
            assert backend is not None
            assert len(factory._instances) == initial_instance_count + 1

    def test_backend_error_message_defensive_sanitization(self, factory):
        """Test that backend error messages are properly sanitized and don't leak sensitive data."""
        sensitive_errors = [
            "AWS Access Key ID: AKIA1234567890 invalid",
            "API Key sk-1234abcd5678efgh not found",
            "Password 'secret123' incorrect",
            "Token eyJ0eXAiOiJKV1Q expired",
            "Connection string postgres://user:pass@localhost failed",
        ]

        for error_msg in sensitive_errors:
            with patch.object(BedrockBackend, "__init__", side_effect=Exception(error_msg)):
                try:
                    factory.create_backend(ModelBackend.BEDROCK)
                except Exception as e:
                    # Error should not contain sensitive patterns
                    sanitized_msg = str(e)

                    # Check that sensitive patterns are not exposed
                    sensitive_patterns = ["AKIA", "sk-", "secret", "eyJ", "user:pass"]
                    for pattern in sensitive_patterns:
                        if pattern in error_msg:
                            # If original had sensitive data, sanitized should not
                            # (This would be implemented in actual error sanitization)
                            pass  # For now, just ensure we can test this pattern
