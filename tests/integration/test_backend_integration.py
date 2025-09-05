"""
Integration tests for LLM backend connectivity and response validation.

These tests verify:
- Real backend connectivity (when credentials available)
- Response parsing and validation
- Error handling and fallback behavior
- Backend-specific configuration
"""

import asyncio
import os
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from decision_matrix_mcp.backends.bedrock import BedrockBackend
from decision_matrix_mcp.backends.factory import BackendFactory
from decision_matrix_mcp.backends.litellm import LiteLLMBackend
from decision_matrix_mcp.backends.ollama import OllamaBackend
from decision_matrix_mcp.exceptions import LLMAPIError, LLMBackendError, LLMConfigurationError
from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend


# Test fixtures
@pytest.fixture()
def sample_criterion():
    """Create a sample criterion for testing."""
    return Criterion(
        name="performance",
        description="Evaluates system performance characteristics",
        weight=1.0,
        system_prompt="You are evaluating performance. Respond with a JSON object containing 'score' (0-10) and 'justification'.",
        model_backend=ModelBackend.BEDROCK,
        temperature=0.1,
        max_tokens=256,
    )


@pytest.fixture()
def sample_criterion_thread(sample_criterion):
    """Create a sample criterion thread for testing."""
    thread = CriterionThread(id=str(uuid4()), criterion=sample_criterion)
    thread.add_message("user", "Please evaluate PostgreSQL for performance.")
    return thread


class TestBackendFactory:
    """Test backend factory functionality."""

    def test_factory_creates_bedrock_backend(self):
        """Test factory creates Bedrock backend."""
        factory = BackendFactory()
        backend = factory.create_backend(ModelBackend.BEDROCK)
        assert isinstance(backend, BedrockBackend)

    def test_factory_creates_litellm_backend(self):
        """Test factory creates LiteLLM backend."""
        factory = BackendFactory()
        backend = factory.create_backend(ModelBackend.LITELLM)
        assert isinstance(backend, LiteLLMBackend)

    def test_factory_creates_ollama_backend(self):
        """Test factory creates Ollama backend."""
        backend = BackendFactory().create_backend(ModelBackend.OLLAMA)
        assert isinstance(backend, OllamaBackend)

    def test_factory_raises_for_invalid_backend(self):
        """Test factory raises error for invalid backend type."""
        from decision_matrix_mcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown model backend"):
            BackendFactory().create_backend("invalid_backend")


class TestBedrockBackendIntegration:
    """Test Bedrock backend integration."""

    @pytest.fixture()
    def bedrock_backend(self):
        """Create Bedrock backend for testing."""
        return BedrockBackend()

    def test_bedrock_backend_initialization(self, bedrock_backend):
        """Test Bedrock backend initializes correctly."""
        assert bedrock_backend.name == "bedrock"
        assert bedrock_backend.supports_streaming is False

    @pytest.mark.asyncio()
    async def test_bedrock_generate_with_mock(self, bedrock_backend, sample_criterion_thread):
        """Test Bedrock generate_response method with mocked response."""
        mock_bedrock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '{"score": 8.5, "justification": "PostgreSQL shows excellent performance characteristics with strong indexing and query optimization."}'
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }

        # Mock the boto3 client and its converse method
        mock_client = Mock()
        mock_client.converse.return_value = mock_bedrock_response

        with patch.object(bedrock_backend, "_get_bedrock_client", return_value=mock_client):
            result = await bedrock_backend.generate_response(sample_criterion_thread)

            assert "8.5" in result
            assert "PostgreSQL" in result
            assert "performance" in result.lower()
            mock_client.converse.assert_called_once()

    @pytest.mark.skipif(not os.environ.get("AWS_PROFILE"), reason="AWS credentials not available")
    @pytest.mark.asyncio()
    async def test_bedrock_real_connection(self, bedrock_backend, sample_criterion_thread):
        """Test real Bedrock connection (only if AWS credentials available)."""
        try:
            result = await bedrock_backend.generate_response(sample_criterion_thread)

            assert isinstance(result, str)
            assert len(result) > 0
            # Should contain some form of evaluation content
            # Note: Real responses may vary, so we just check basic structure

        except (LLMBackendError, LLMConfigurationError, LLMAPIError) as e:
            # If we get a backend error, that's expected in some environments
            pytest.skip(f"Bedrock not available: {e}")

    @pytest.mark.asyncio()
    async def test_bedrock_error_handling(self, bedrock_backend, sample_criterion_thread):
        """Test Bedrock error handling."""
        from botocore.exceptions import ClientError

        # Mock client that raises an error
        mock_client = Mock()
        mock_client.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "ServiceException", "Message": "Connection timeout"}},
            operation_name="Converse",
        )

        with (
            patch.object(bedrock_backend, "_get_bedrock_client", return_value=mock_client),
            pytest.raises(LLMAPIError, match="Bedrock API call failed"),
        ):
            await bedrock_backend.generate_response(sample_criterion_thread)

    @pytest.mark.asyncio()
    async def test_bedrock_response_parsing(self, bedrock_backend, sample_criterion_thread):
        """Test Bedrock response parsing with various formats."""
        test_cases = [
            {
                "name": "valid_response",
                "response": {
                    "output": {"message": {"content": [{"text": "8.0: Good performance"}]}},
                    "usage": {"inputTokens": 10, "outputTokens": 5},
                },
                "expected_content": "8.0: Good performance",
            },
            {
                "name": "empty_content",
                "response": {
                    "output": {"message": {"content": [{"text": ""}]}},
                    "usage": {"inputTokens": 10, "outputTokens": 1},
                },
                "expected_content": "",  # Empty text should be returned as is
            },
            {
                "name": "no_content_blocks",
                "response": {
                    "output": {"message": {"content": []}},
                    "usage": {"inputTokens": 10, "outputTokens": 0},
                },
                "expected_error": "LLMAPIError",  # Should raise error for empty content list
            },
            {
                "name": "malformed_response",
                "response": {},
                "expected_error": "LLMAPIError",  # Should raise error for missing output
            },
        ]

        for case in test_cases:
            mock_client = Mock()
            mock_client.converse.return_value = case["response"]

            with patch.object(bedrock_backend, "_get_bedrock_client", return_value=mock_client):
                if "expected_error" in case:
                    with pytest.raises(LLMAPIError):
                        await bedrock_backend.generate_response(sample_criterion_thread)
                else:
                    result = await bedrock_backend.generate_response(sample_criterion_thread)
                    assert case["expected_content"] in result


class TestLiteLLMBackendIntegration:
    """Test LiteLLM backend integration."""

    @pytest.fixture()
    def litellm_backend(self):
        """Create LiteLLM backend for testing."""
        return LiteLLMBackend()

    def test_litellm_backend_initialization(self, litellm_backend):
        """Test LiteLLM backend initializes correctly."""
        assert litellm_backend.name == "litellm"
        assert litellm_backend.supports_streaming is False

    @pytest.mark.asyncio()
    async def test_litellm_generate_with_mock(self, litellm_backend, sample_criterion_thread):
        """Test LiteLLM generate_response method with mocked response."""
        # Create a mock response that looks like a real LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            '{"score": 7.5, "justification": "PostgreSQL has moderate performance characteristics with good optimization potential."}'
        )

        with patch(
            "decision_matrix_mcp.backends.litellm.litellm.acompletion", return_value=mock_response
        ):
            result = await litellm_backend.generate_response(sample_criterion_thread)

            assert "7.5" in result
            assert "PostgreSQL" in result
            assert "performance" in result.lower()

    @pytest.mark.skipif(
        not os.environ.get("LITELLM_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
        reason="LiteLLM API key not available",
    )
    @pytest.mark.asyncio()
    async def test_litellm_real_connection(self, litellm_backend, sample_criterion_thread):
        """Test real LiteLLM connection (only if API key available)."""
        try:
            result = await litellm_backend.generate_response(sample_criterion_thread)

            assert isinstance(result, str)
            assert len(result) > 0
            # Should contain some form of evaluation content

        except (LLMBackendError, LLMConfigurationError, LLMAPIError) as e:
            pytest.skip(f"LiteLLM not available: {e}")

    @pytest.mark.asyncio()
    async def test_litellm_error_handling(self, litellm_backend, sample_criterion_thread):
        """Test LiteLLM error handling."""
        with patch(
            "decision_matrix_mcp.backends.litellm.litellm.acompletion",
            side_effect=Exception("API rate limit exceeded"),
        ):
            with pytest.raises(LLMAPIError, match="LiteLLM API call failed"):
                await litellm_backend.generate_response(sample_criterion_thread)

    @pytest.mark.asyncio()
    async def test_litellm_response_validation(self, litellm_backend, sample_criterion_thread):
        """Test LiteLLM response validation."""

        # Test various response formats
        test_cases = [
            {
                "name": "valid_response",
                "response": Mock(choices=[Mock(message=Mock(content="Valid response"))]),
                "expected_content": "Valid response",
            },
            {
                "name": "empty_content",
                "response": Mock(choices=[Mock(message=Mock(content=""))]),
                "expected_content": "",  # Empty content should be returned as-is
            },
            {
                "name": "no_choices",
                "response": Mock(choices=[]),
                "expected_error": "LLMAPIError",  # Should raise IndexError -> LLMAPIError
            },
            {
                "name": "none_content",
                "response": Mock(choices=[Mock(message=Mock(content=None))]),
                "expected_none": True,  # None content returns None directly
            },
        ]

        for case in test_cases:
            with patch(
                "decision_matrix_mcp.backends.litellm.litellm.acompletion",
                return_value=case["response"],
            ):
                if "expected_error" in case:
                    with pytest.raises(LLMAPIError):
                        await litellm_backend.generate_response(sample_criterion_thread)
                elif "expected_none" in case:
                    result = await litellm_backend.generate_response(sample_criterion_thread)
                    assert result is None
                else:
                    result = await litellm_backend.generate_response(sample_criterion_thread)
                    assert case["expected_content"] in result


class TestOllamaBackendIntegration:
    """Test Ollama backend integration."""

    @pytest.fixture()
    def ollama_backend(self):
        """Create Ollama backend for testing."""
        return OllamaBackend()

    def test_ollama_backend_initialization(self, ollama_backend):
        """Test Ollama backend initializes correctly."""
        assert ollama_backend.name == "ollama"
        assert ollama_backend.supports_streaming is False

    @pytest.mark.asyncio()
    async def test_ollama_generate_with_mock(self, ollama_backend, sample_criterion_thread):
        """Test Ollama generate_response method with mocked response."""
        mock_response_json = {
            "message": {
                "role": "assistant",
                "content": '{"score": 6.0, "justification": "PostgreSQL has moderate performance characteristics."}',
            },
            "done": True,
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_json
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        # Mock the httpx AsyncClient post method
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await ollama_backend.generate_response(sample_criterion_thread)

            assert "6.0" in result
            assert "PostgreSQL" in result
            assert "performance" in result.lower()

    def test_ollama_availability_check(self, ollama_backend):
        """Test Ollama backend availability checking."""
        # The is_available method just checks if httpx is installed
        # Since we have httpx in our test environment, it should return True
        assert ollama_backend.is_available() is True

    @pytest.mark.asyncio()
    async def test_ollama_error_handling(self, ollama_backend, sample_criterion_thread):
        """Test Ollama error handling."""
        with patch("httpx.AsyncClient.post", side_effect=Exception("Connection timeout")):
            with pytest.raises(LLMAPIError, match="Ollama call failed"):
                await ollama_backend.generate_response(sample_criterion_thread)

    @pytest.mark.asyncio()
    async def test_ollama_response_parsing(self, ollama_backend, sample_criterion_thread):
        """Test Ollama response parsing with various formats."""
        test_cases = [
            {
                "name": "valid_response",
                "response_data": {
                    "message": {"role": "assistant", "content": "Valid response"},
                    "done": True,
                },
                "status_code": 200,
                "expected_content": "Valid response",
            },
            {
                "name": "empty_content",
                "response_data": {"message": {"role": "assistant", "content": ""}, "done": True},
                "status_code": 200,
                "expected_content": "",
            },
            {
                "name": "missing_message",
                "response_data": {"done": True},
                "status_code": 200,
                "expected_error": "LLMAPIError",  # KeyError -> LLMAPIError
            },
            {
                "name": "model_not_found",
                "response_data": {"error": "Model not found"},
                "status_code": 404,
                "expected_error": "LLMAPIError",  # 404 -> specific error
            },
        ]

        for case in test_cases:
            mock_response = Mock()
            mock_response.json.return_value = case["response_data"]
            mock_response.status_code = case["status_code"]

            with patch("httpx.AsyncClient.post", return_value=mock_response):
                if "expected_error" in case:
                    with pytest.raises(LLMAPIError):
                        await ollama_backend.generate_response(sample_criterion_thread)
                else:
                    result = await ollama_backend.generate_response(sample_criterion_thread)
                    assert case["expected_content"] in result


class TestBackendContractValidation:
    """Test that all backends conform to the expected contract."""

    @pytest.mark.parametrize(
        "backend_type",
        [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA],
    )
    def test_backend_implements_interface(self, backend_type):
        """Test that each backend implements the required interface."""
        backend = BackendFactory().create_backend(backend_type)

        # Check required attributes
        assert hasattr(backend, "name")
        assert hasattr(backend, "supports_streaming")
        assert isinstance(backend.name, str)
        assert isinstance(backend.supports_streaming, bool)

        # Check required methods
        assert hasattr(backend, "generate_response")
        assert callable(backend.generate_response)
        assert hasattr(backend, "is_available")
        assert callable(backend.is_available)

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        "backend_type",
        [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA],
    )
    async def test_backend_generate_signature(self, backend_type, sample_criterion_thread):
        """Test that generate_response method has consistent signature across backends."""
        backend = BackendFactory().create_backend(backend_type)

        # Create appropriate mocks for each backend type
        if backend_type == ModelBackend.BEDROCK:
            mock_response = {
                "output": {"message": {"content": [{"text": "Test response"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5},
            }
            mock_client = Mock()
            mock_client.converse.return_value = mock_response

            with patch.object(backend, "_get_bedrock_client", return_value=mock_client):
                result = await backend.generate_response(sample_criterion_thread)

        elif backend_type == ModelBackend.LITELLM:
            mock_response = Mock(choices=[Mock(message=Mock(content="Test response"))])

            with patch(
                "decision_matrix_mcp.backends.litellm.litellm.acompletion",
                return_value=mock_response,
            ):
                result = await backend.generate_response(sample_criterion_thread)

        else:  # OLLAMA
            mock_response = Mock()
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": "Test response"},
                "done": True,
            }
            mock_response.status_code = 200

            with patch("httpx.AsyncClient.post", return_value=mock_response):
                result = await backend.generate_response(sample_criterion_thread)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "backend_type",
        [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA],
    )
    def test_backend_availability_check(self, backend_type):
        """Test that is_available method works for all backends."""
        backend = BackendFactory().create_backend(backend_type)

        # The is_available method is synchronous and just checks for dependencies
        is_available = backend.is_available()
        assert isinstance(is_available, bool)

        # In our test environment, all dependencies should be available
        # (boto3 is optional but httpx and litellm are available)
        if backend_type == ModelBackend.BEDROCK:
            # May be True or False depending on whether boto3 is installed
            assert isinstance(is_available, bool)
        else:
            # LiteLLM and Ollama dependencies should be available in test environment
            assert is_available is True


class TestBackendErrorRecovery:
    """Test backend error recovery and fallback mechanisms."""

    @pytest.mark.asyncio()
    async def test_backend_timeout_handling(self, sample_criterion_thread):
        """Test that backends handle timeouts gracefully."""
        backends = [
            BackendFactory().create_backend(ModelBackend.BEDROCK),
            BackendFactory().create_backend(ModelBackend.LITELLM),
            BackendFactory().create_backend(ModelBackend.OLLAMA),
        ]

        for backend in backends:
            # Mock timeout exception
            if backend.name == "bedrock":
                from botocore.exceptions import ClientError

                timeout_error = ClientError(
                    error_response={
                        "Error": {"Code": "TimeoutError", "Message": "Request timed out"}
                    },
                    operation_name="Converse",
                )
                mock_client = Mock()
                mock_client.converse.side_effect = timeout_error

                with (
                    patch.object(backend._backend, "_get_bedrock_client", return_value=mock_client),
                    pytest.raises((LLMAPIError, LLMBackendError)),
                ):
                    await backend.generate_response(sample_criterion_thread)

            elif backend.name == "litellm":
                with patch(
                    "decision_matrix_mcp.backends.litellm.litellm.acompletion",
                    side_effect=asyncio.TimeoutError(),
                ):
                    with pytest.raises((LLMAPIError, LLMBackendError)):
                        await backend.generate_response(sample_criterion_thread)

            elif backend.name == "ollama":
                with patch("httpx.AsyncClient.post", side_effect=asyncio.TimeoutError()):
                    with pytest.raises((LLMAPIError, LLMBackendError)):
                        await backend.generate_response(sample_criterion_thread)

    @pytest.mark.asyncio()
    async def test_backend_rate_limit_handling(self, sample_criterion_thread):
        """Test that backends handle rate limits appropriately."""
        backends = [
            BackendFactory().create_backend(ModelBackend.BEDROCK),
            BackendFactory().create_backend(ModelBackend.LITELLM),
            BackendFactory().create_backend(ModelBackend.OLLAMA),
        ]

        for backend in backends:
            if backend.name == "bedrock":
                from botocore.exceptions import ClientError

                rate_limit_error = ClientError(
                    error_response={
                        "Error": {"Code": "ThrottlingException", "Message": "Rate limit exceeded"}
                    },
                    operation_name="Converse",
                )
                mock_client = Mock()
                mock_client.converse.side_effect = rate_limit_error

                with patch.object(
                    backend._backend, "_get_bedrock_client", return_value=mock_client
                ):
                    with pytest.raises((LLMAPIError, LLMBackendError)):
                        await backend.generate_response(sample_criterion_thread)

            elif backend.name == "litellm":
                rate_limit_error = Exception("Rate limit exceeded")
                with patch(
                    "decision_matrix_mcp.backends.litellm.litellm.acompletion",
                    side_effect=rate_limit_error,
                ):
                    with pytest.raises((LLMAPIError, LLMBackendError)):
                        await backend.generate_response(sample_criterion_thread)

            elif backend.name == "ollama":
                rate_limit_error = Exception("Rate limit exceeded")
                with patch("httpx.AsyncClient.post", side_effect=rate_limit_error):
                    with pytest.raises((LLMAPIError, LLMBackendError)):
                        await backend.generate_response(sample_criterion_thread)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
