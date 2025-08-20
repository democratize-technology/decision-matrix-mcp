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

import pytest

from decision_matrix_mcp.backends.bedrock import BedrockBackend
from decision_matrix_mcp.backends.factory import BackendFactory
from decision_matrix_mcp.backends.litellm import LiteLLMBackend
from decision_matrix_mcp.backends.ollama import OllamaBackend
from decision_matrix_mcp.exceptions import LLMBackendError
from decision_matrix_mcp.models import ModelBackend


class TestBackendFactory:
    """Test backend factory functionality."""

    def test_factory_creates_bedrock_backend(self):
        """Test factory creates Bedrock backend."""
        backend = BackendFactory.create_backend(ModelBackend.BEDROCK)
        assert isinstance(backend, BedrockBackend)

    def test_factory_creates_litellm_backend(self):
        """Test factory creates LiteLLM backend."""
        backend = BackendFactory.create_backend(ModelBackend.LITELLM)
        assert isinstance(backend, LiteLLMBackend)

    def test_factory_creates_ollama_backend(self):
        """Test factory creates Ollama backend."""
        backend = BackendFactory.create_backend(ModelBackend.OLLAMA)
        assert isinstance(backend, OllamaBackend)

    def test_factory_raises_for_invalid_backend(self):
        """Test factory raises error for invalid backend type."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            BackendFactory.create_backend("invalid_backend")


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
    async def test_bedrock_generate_with_mock(self, bedrock_backend):
        """Test Bedrock generate method with mocked response."""
        mock_response = {
            "content": [{"text": "8.5: This option shows excellent performance characteristics."}],
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }

        with patch.object(bedrock_backend, "_make_bedrock_request", return_value=mock_response):
            result = await bedrock_backend.generate(
                system_prompt="You are evaluating performance.",
                user_prompt="Rate PostgreSQL for performance.",
                temperature=0.1,
                max_tokens=1024,
            )

            assert "8.5" in result
            assert "performance" in result.lower()

    @pytest.mark.skipif(not os.environ.get("AWS_PROFILE"), reason="AWS credentials not available")
    @pytest.mark.asyncio()
    async def test_bedrock_real_connection(self, bedrock_backend):
        """Test real Bedrock connection (only if AWS credentials available)."""
        try:
            result = await bedrock_backend.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'test successful' if you can respond.",
                temperature=0.0,
                max_tokens=50,
            )

            assert isinstance(result, str)
            assert len(result) > 0
            # Should contain some form of success indicator
            assert any(word in result.lower() for word in ["test", "successful", "yes", "respond"])

        except LLMBackendError as e:
            # If we get a specific backend error, that's expected in some environments
            pytest.skip(f"Bedrock not available: {e}")

    @pytest.mark.asyncio()
    async def test_bedrock_error_handling(self, bedrock_backend):
        """Test Bedrock error handling."""
        with (
            patch.object(
                bedrock_backend,
                "_make_bedrock_request",
                side_effect=Exception("Connection timeout"),
            ),
            pytest.raises(LLMBackendError, match="Bedrock API error"),
        ):
            await bedrock_backend.generate(
                system_prompt="Test",
                user_prompt="Test",
                temperature=0.1,
            )

    @pytest.mark.asyncio()
    async def test_bedrock_response_parsing(self, bedrock_backend):
        """Test Bedrock response parsing with various formats."""
        test_cases = [
            {
                "response": {"content": [{"text": "8.0: Good performance"}]},
                "expected_content": "8.0: Good performance",
            },
            {"response": {"content": [{"text": ""}]}, "expected_error": "Empty response"},
            {"response": {"content": []}, "expected_error": "No content blocks"},
            {"response": {}, "expected_error": "content"},
        ]

        for case in test_cases:
            with patch.object(
                bedrock_backend,
                "_make_bedrock_request",
                return_value=case["response"],
            ):
                if "expected_error" in case:
                    with pytest.raises(LLMBackendError, match=case["expected_error"]):
                        await bedrock_backend.generate("system", "user")
                else:
                    result = await bedrock_backend.generate("system", "user")
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
    async def test_litellm_generate_with_mock(self, litellm_backend):
        """Test LiteLLM generate method with mocked response."""
        from litellm import ModelResponse

        # Create a mock response that looks like a real LiteLLM response
        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            "7.5: This framework has a moderate learning curve."
        )

        with patch("litellm.acompletion", return_value=mock_response):
            result = await litellm_backend.generate(
                system_prompt="You are evaluating learning curves.",
                user_prompt="Rate React's learning curve.",
                model="gpt-3.5-turbo",
                temperature=0.2,
            )

            assert "7.5" in result
            assert "learning curve" in result.lower()

    @pytest.mark.skipif(
        not os.environ.get("LITELLM_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
        reason="LiteLLM API key not available",
    )
    @pytest.mark.asyncio()
    async def test_litellm_real_connection(self, litellm_backend):
        """Test real LiteLLM connection (only if API key available)."""
        try:
            result = await litellm_backend.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Respond with exactly: 'Connection successful'",
                model="gpt-3.5-turbo",
                temperature=0.0,
                max_tokens=20,
            )

            assert isinstance(result, str)
            assert len(result) > 0
            # Should contain success indicator
            assert "successful" in result.lower() or "connection" in result.lower()

        except LLMBackendError as e:
            pytest.skip(f"LiteLLM not available: {e}")

    @pytest.mark.asyncio()
    async def test_litellm_error_handling(self, litellm_backend):
        """Test LiteLLM error handling."""
        with patch("litellm.acompletion", side_effect=Exception("API rate limit exceeded")):
            with pytest.raises(LLMBackendError, match="LiteLLM API error"):
                await litellm_backend.generate(
                    system_prompt="Test",
                    user_prompt="Test",
                    model="gpt-3.5-turbo",
                )

    @pytest.mark.asyncio()
    async def test_litellm_response_validation(self, litellm_backend):
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
                "expected_error": "Empty response",
            },
            {
                "name": "no_choices",
                "response": Mock(choices=[]),
                "expected_error": "No response choices",
            },
            {
                "name": "none_content",
                "response": Mock(choices=[Mock(message=Mock(content=None))]),
                "expected_error": "Empty response",
            },
        ]

        for case in test_cases:
            with patch("litellm.acompletion", return_value=case["response"]):
                if "expected_error" in case:
                    with pytest.raises(LLMBackendError, match=case["expected_error"]):
                        await litellm_backend.generate("system", "user", model="gpt-3.5-turbo")
                else:
                    result = await litellm_backend.generate("system", "user", model="gpt-3.5-turbo")
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
        assert ollama_backend.base_url == "http://localhost:11434"

    @pytest.mark.asyncio()
    async def test_ollama_generate_with_mock(self, ollama_backend):
        """Test Ollama generate method with mocked response."""
        mock_response_data = {
            "response": "6.0: This option has moderate cost efficiency.",
            "done": True,
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await ollama_backend.generate(
                system_prompt="You are evaluating cost efficiency.",
                user_prompt="Rate MongoDB's cost efficiency.",
                model="llama3.2:3b",
                temperature=0.3,
            )

            assert "6.0" in result
            assert "cost" in result.lower()

    @pytest.mark.asyncio()
    async def test_ollama_connection_check(self, ollama_backend):
        """Test Ollama connection checking."""
        # Test successful connection
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            is_available = await ollama_backend.is_available()
            assert is_available is True

        # Test failed connection
        with patch("httpx.AsyncClient.get", side_effect=Exception("Connection refused")):
            is_available = await ollama_backend.is_available()
            assert is_available is False

    @pytest.mark.asyncio()
    async def test_ollama_error_handling(self, ollama_backend):
        """Test Ollama error handling."""
        with patch("httpx.AsyncClient.post", side_effect=Exception("Connection timeout")):
            with pytest.raises(LLMBackendError, match="Ollama API error"):
                await ollama_backend.generate(
                    system_prompt="Test",
                    user_prompt="Test",
                    model="llama3.2:3b",
                )

    @pytest.mark.asyncio()
    async def test_ollama_response_parsing(self, ollama_backend):
        """Test Ollama response parsing with various formats."""
        test_cases = [
            {
                "response_data": {"response": "Valid response", "done": True},
                "expected_content": "Valid response",
            },
            {"response_data": {"response": "", "done": True}, "expected_error": "Empty response"},
            {"response_data": {"done": True}, "expected_error": "No response content"},
            {
                "response_data": {"response": "Incomplete"},
                "expected_error": "Response not complete",
            },
        ]

        for case in test_cases:
            mock_response = Mock()
            mock_response.json.return_value = case["response_data"]
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.post", return_value=mock_response):
                if "expected_error" in case:
                    with pytest.raises(LLMBackendError, match=case["expected_error"]):
                        await ollama_backend.generate("system", "user", model="llama3.2:3b")
                else:
                    result = await ollama_backend.generate("system", "user", model="llama3.2:3b")
                    assert case["expected_content"] in result


class TestBackendContractValidation:
    """Test that all backends conform to the expected contract."""

    @pytest.mark.parametrize(
        "backend_type",
        [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA],
    )
    def test_backend_implements_interface(self, backend_type):
        """Test that each backend implements the required interface."""
        backend = BackendFactory.create_backend(backend_type)

        # Check required attributes
        assert hasattr(backend, "name")
        assert hasattr(backend, "supports_streaming")
        assert isinstance(backend.name, str)
        assert isinstance(backend.supports_streaming, bool)

        # Check required methods
        assert hasattr(backend, "generate")
        assert callable(backend.generate)
        assert hasattr(backend, "is_available")
        assert callable(backend.is_available)

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        "backend_type",
        [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA],
    )
    async def test_backend_generate_signature(self, backend_type):
        """Test that generate method has consistent signature across backends."""
        backend = BackendFactory.create_backend(backend_type)

        # Create appropriate mocks for each backend type
        if backend_type == ModelBackend.BEDROCK:
            mock_response = {"content": [{"text": "Test response"}]}
            mock_method = "_make_bedrock_request"
        elif backend_type == ModelBackend.LITELLM:
            mock_response = Mock(choices=[Mock(message=Mock(content="Test response"))])
            mock_method = None  # Use litellm.acompletion directly
        else:  # OLLAMA
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Test response", "done": True}
            mock_response.raise_for_status.return_value = None
            mock_method = None  # Use httpx.AsyncClient.post directly

        # Test that generate method can be called with required parameters
        if mock_method:
            with patch.object(backend, mock_method, return_value=mock_response):
                result = await backend.generate(
                    system_prompt="Test system prompt",
                    user_prompt="Test user prompt",
                )
        elif backend_type == ModelBackend.LITELLM:
            with patch("litellm.acompletion", return_value=mock_response):
                result = await backend.generate(
                    system_prompt="Test system prompt",
                    user_prompt="Test user prompt",
                    model="gpt-3.5-turbo",
                )
        else:  # OLLAMA
            with patch("httpx.AsyncClient.post", return_value=mock_response):
                result = await backend.generate(
                    system_prompt="Test system prompt",
                    user_prompt="Test user prompt",
                    model="llama3.2:3b",
                )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        "backend_type",
        [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA],
    )
    async def test_backend_availability_check(self, backend_type):
        """Test that is_available method works for all backends."""
        backend = BackendFactory.create_backend(backend_type)

        # Mock successful availability check
        if backend_type == ModelBackend.BEDROCK:
            with patch.object(backend, "_check_bedrock_connection", return_value=True):
                is_available = await backend.is_available()
                assert isinstance(is_available, bool)
        elif backend_type == ModelBackend.LITELLM:
            with patch("litellm.acompletion") as mock_completion:
                mock_response = Mock(choices=[Mock(message=Mock(content="test"))])
                mock_completion.return_value = mock_response
                is_available = await backend.is_available()
                assert isinstance(is_available, bool)
        else:  # OLLAMA
            with patch("httpx.AsyncClient.get") as mock_get:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                is_available = await backend.is_available()
                assert isinstance(is_available, bool)


class TestBackendErrorRecovery:
    """Test backend error recovery and fallback mechanisms."""

    @pytest.mark.asyncio()
    async def test_backend_timeout_handling(self):
        """Test that backends handle timeouts gracefully."""
        backends = [
            BackendFactory.create_backend(ModelBackend.BEDROCK),
            BackendFactory.create_backend(ModelBackend.LITELLM),
            BackendFactory.create_backend(ModelBackend.OLLAMA),
        ]

        for backend in backends:
            # Mock timeout exception
            if backend.name == "bedrock":
                with (
                    patch.object(
                        backend,
                        "_make_bedrock_request",
                        side_effect=asyncio.TimeoutError(),
                    ),
                    pytest.raises(LLMBackendError, match="timeout|Bedrock API error"),
                ):
                    await backend.generate("system", "user")
            elif backend.name == "litellm":
                with patch("litellm.acompletion", side_effect=asyncio.TimeoutError()):
                    with pytest.raises(LLMBackendError, match="timeout|LiteLLM API error"):
                        await backend.generate("system", "user", model="gpt-3.5-turbo")
            elif backend.name == "ollama":
                with patch("httpx.AsyncClient.post", side_effect=asyncio.TimeoutError()):
                    with pytest.raises(LLMBackendError, match="timeout|Ollama API error"):
                        await backend.generate("system", "user", model="llama3.2:3b")

    @pytest.mark.asyncio()
    async def test_backend_rate_limit_handling(self):
        """Test that backends handle rate limits appropriately."""
        backends = [
            BackendFactory.create_backend(ModelBackend.BEDROCK),
            BackendFactory.create_backend(ModelBackend.LITELLM),
            BackendFactory.create_backend(ModelBackend.OLLAMA),
        ]

        for backend in backends:
            rate_limit_error = Exception("Rate limit exceeded")

            if backend.name == "bedrock":
                with patch.object(backend, "_make_bedrock_request", side_effect=rate_limit_error):
                    with pytest.raises(LLMBackendError):
                        await backend.generate("system", "user")
            elif backend.name == "litellm":
                with patch("litellm.acompletion", side_effect=rate_limit_error):
                    with pytest.raises(LLMBackendError):
                        await backend.generate("system", "user", model="gpt-3.5-turbo")
            elif backend.name == "ollama":
                with patch("httpx.AsyncClient.post", side_effect=rate_limit_error):
                    with pytest.raises(LLMBackendError):
                        await backend.generate("system", "user", model="llama3.2:3b")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
