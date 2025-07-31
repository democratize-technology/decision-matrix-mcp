"""Tests for enhanced error handling in DecisionOrchestrator"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decision_matrix_mcp.exceptions import (
    ConfigurationError,
    LLMAPIError,
    LLMBackendError,
    LLMConfigurationError,
)
from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestOrchestratorErrorHandling:
    """Test enhanced error handling in the DecisionOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance"""
        return DecisionOrchestrator(max_retries=2, retry_delay=0.1)

    @pytest.fixture
    def bedrock_thread(self):
        """Create a thread with Bedrock backend"""
        criterion = Criterion(
            name="Performance",
            description="Evaluate performance",
            weight=1.0,
            model_backend=ModelBackend.BEDROCK,
        )
        return CriterionThread(id="test-thread", criterion=criterion)

    @pytest.fixture
    def litellm_thread(self):
        """Create a thread with LiteLLM backend"""
        criterion = Criterion(
            name="Cost",
            description="Evaluate cost",
            weight=1.0,
            model_backend=ModelBackend.LITELLM,
        )
        return CriterionThread(id="test-thread", criterion=criterion)

    @pytest.fixture
    def ollama_thread(self):
        """Create a thread with Ollama backend"""
        criterion = Criterion(
            name="Quality",
            description="Evaluate quality",
            weight=1.0,
            model_backend=ModelBackend.OLLAMA,
        )
        return CriterionThread(id="test-thread", criterion=criterion)

    @pytest.mark.asyncio
    async def test_bedrock_configuration_error(self, orchestrator, bedrock_thread):
        """Test Bedrock configuration error handling"""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._call_bedrock(bedrock_thread)

            error = exc_info.value
            assert error.backend == "bedrock"
            assert "boto3 dependency missing" in str(error)
            assert error.user_message == "bedrock backend not properly configured"
            assert isinstance(error.original_error, ImportError)

    @pytest.mark.asyncio
    async def test_bedrock_rate_limit_error(self, orchestrator, bedrock_thread):
        """Test Bedrock rate limit error handling"""
        from botocore.exceptions import ClientError

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = ClientError(
                {
                    "Error": {
                        "Code": "ThrottlingException",
                        "Message": "Rate limit exceeded",
                    }
                },
                "invoke_model",
            )
            mock_session.return_value.client.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_bedrock(bedrock_thread)

            error = exc_info.value
            assert error.backend == "bedrock"
            assert "rate limit" in error.user_message.lower()
            assert isinstance(error.original_error, ClientError)

    @pytest.mark.asyncio
    async def test_bedrock_invalid_model_error(self, orchestrator, bedrock_thread):
        """Test Bedrock invalid model error handling"""
        from botocore.exceptions import ClientError

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Invalid model ID specified",
                    }
                },
                "invoke_model",
            )
            mock_session.return_value.client.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_bedrock(bedrock_thread)

            error = exc_info.value
            assert error.backend == "bedrock"
            assert "Invalid model ID" in error.user_message

    @pytest.mark.asyncio
    async def test_bedrock_unexpected_error(self, orchestrator, bedrock_thread):
        """Test Bedrock unexpected error handling"""
        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = RuntimeError("Unexpected error")
            mock_session.return_value.client.return_value = mock_client

            with pytest.raises(LLMBackendError) as exc_info:
                await orchestrator._call_bedrock(bedrock_thread)

            error = exc_info.value
            assert error.backend == "bedrock"
            assert "Unexpected error in Bedrock call" in str(error)
            assert error.user_message == "An unexpected error occurred"
            assert isinstance(error.original_error, RuntimeError)

    @pytest.mark.asyncio
    async def test_bedrock_invalid_response_format(self, orchestrator, bedrock_thread):
        """Test Bedrock invalid response format error"""
        mock_response = {
            "body": MagicMock(read=lambda: b'{"invalid": "format"}')
        }

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_session.return_value.client.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_bedrock(bedrock_thread)

            error = exc_info.value
            assert error.backend == "bedrock"
            assert "Invalid response format" in str(error)
            assert error.user_message == "Unexpected response format from LLM"

    @pytest.mark.asyncio
    async def test_litellm_configuration_error(self, orchestrator, litellm_thread):
        """Test LiteLLM configuration error handling"""
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._call_litellm(litellm_thread)

            error = exc_info.value
            assert error.backend == "litellm"
            assert "litellm dependency missing" in str(error)

    @pytest.mark.asyncio
    async def test_litellm_api_key_error(self, orchestrator, litellm_thread):
        """Test LiteLLM API key error handling"""
        with patch("litellm.acompletion", side_effect=Exception("Invalid API key")):
            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_litellm(litellm_thread)

            error = exc_info.value
            assert error.backend == "litellm"
            assert "API authentication failed" in error.user_message

    @pytest.mark.asyncio
    async def test_litellm_rate_limit_error(self, orchestrator, litellm_thread):
        """Test LiteLLM rate limit error handling"""
        with patch("litellm.acompletion", side_effect=Exception("Rate limit exceeded")):
            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_litellm(litellm_thread)

            error = exc_info.value
            assert error.backend == "litellm"
            assert "API rate limit exceeded" in error.user_message

    @pytest.mark.asyncio
    async def test_litellm_model_not_found(self, orchestrator, litellm_thread):
        """Test LiteLLM model not found error"""
        with patch("litellm.acompletion", side_effect=Exception("Model 'gpt-5' not found")):
            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_litellm(litellm_thread)

            error = exc_info.value
            assert error.backend == "litellm"
            assert "Model not available" in error.user_message

    @pytest.mark.asyncio
    async def test_ollama_configuration_error(self, orchestrator, ollama_thread):
        """Test Ollama configuration error handling"""
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._call_ollama(ollama_thread)

            error = exc_info.value
            assert error.backend == "ollama"
            assert "httpx dependency missing" in str(error)

    @pytest.mark.asyncio
    async def test_ollama_connection_error(self, orchestrator, ollama_thread):
        """Test Ollama connection error handling"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_ollama(ollama_thread)

            error = exc_info.value
            assert error.backend == "ollama"
            assert "Cannot connect to Ollama service" in error.user_message

    @pytest.mark.asyncio
    async def test_ollama_model_not_found(self, orchestrator, ollama_thread):
        """Test Ollama model not found error"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'llama3' not found"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_ollama(ollama_thread)

            error = exc_info.value
            assert error.backend == "ollama"
            assert "Model not available in Ollama" in error.user_message

    @pytest.mark.asyncio
    async def test_ollama_service_unavailable(self, orchestrator, ollama_thread):
        """Test Ollama service unavailable error"""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_ollama(ollama_thread)

            error = exc_info.value
            assert error.backend == "ollama"
            assert "Ollama service is not running" in error.user_message

    @pytest.mark.asyncio
    async def test_error_propagation_hierarchy(self, orchestrator, bedrock_thread):
        """Test that custom exceptions are properly propagated"""
        # Create a custom LLMBackendError
        custom_error = LLMBackendError(
            backend="bedrock",
            message="Custom error",
            user_message="User-friendly message",
        )

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = custom_error
            mock_session.return_value.client.return_value = mock_client

            # The custom error should be re-raised without modification
            with pytest.raises(LLMBackendError) as exc_info:
                await orchestrator._call_bedrock(bedrock_thread)

            assert exc_info.value is custom_error

    @pytest.mark.asyncio
    async def test_retry_logic_with_specific_errors(self, orchestrator, bedrock_thread):
        """Test that retry logic handles specific error types correctly"""
        orchestrator.max_retries = 3

        # Test non-retryable errors (should fail immediately)
        non_retryable = LLMConfigurationError(
            backend="bedrock",
            message="Missing API key",
            user_message="Configuration error",
        )

        async def mock_backend_non_retryable(thread):
            raise non_retryable

        orchestrator.backends[ModelBackend.BEDROCK] = mock_backend_non_retryable

        with patch("asyncio.sleep"):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._get_thread_response(bedrock_thread)

            assert exc_info.value is non_retryable

        # Test retryable errors (should retry)
        call_count = 0

        async def mock_backend_retryable(thread):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMAPIError(
                    backend="bedrock",
                    message="Rate limit",
                    user_message="Try again",
                )
            return "Success"

        orchestrator.backends[ModelBackend.BEDROCK] = mock_backend_retryable

        with patch("asyncio.sleep"):
            response = await orchestrator._get_thread_response(bedrock_thread)
            assert response == "Success"
            assert call_count == 2

    def test_exception_attributes(self):
        """Test that exception attributes are properly set"""
        # Test LLMBackendError
        error = LLMBackendError(
            backend="test",
            message="Internal error",
            user_message="User error",
            original_error=ValueError("Original"),
        )
        assert error.backend == "test"
        assert str(error) == "Internal error"
        assert error.user_message == "User error"
        assert isinstance(error.original_error, ValueError)

        # Test LLMConfigurationError
        config_error = LLMConfigurationError(
            backend="test2",
            message="Config error",
        )
        assert config_error.backend == "test2"
        assert config_error.user_message == "test2 backend not properly configured"

        # Test LLMAPIError
        api_error = LLMAPIError(
            backend="test3",
            message="API error",
        )
        assert api_error.backend == "test3"
        assert api_error.user_message == "LLM service temporarily unavailable"