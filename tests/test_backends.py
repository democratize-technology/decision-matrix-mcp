"""Tests for individual backend implementations"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decision_matrix_mcp.backends import BedrockBackend, LiteLLMBackend, OllamaBackend
from decision_matrix_mcp.exceptions import LLMAPIError, LLMConfigurationError
from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend


class TestBedrockBackend:
    """Test BedrockBackend implementation"""

    @pytest.fixture()
    def backend(self):
        """Create a BedrockBackend instance"""
        return BedrockBackend()

    @pytest.fixture()
    def sample_thread(self):
        """Create a sample criterion thread"""
        criterion = Criterion(
            name="Performance",
            description="Evaluate performance",
            weight=2.0,
            model_backend=ModelBackend.BEDROCK,
        )
        thread = CriterionThread(id="test-thread", criterion=criterion)
        thread.add_message("user", "Test message")
        return thread

    def test_is_available_with_boto3(self, backend):
        """Test availability checking when boto3 is available"""
        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", True):
            assert backend.is_available() is True

    def test_is_available_without_boto3(self, backend):
        """Test availability checking when boto3 is not available"""
        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", False):
            assert backend.is_available() is False

    @pytest.mark.asyncio()
    async def test_generate_response_success(self, backend, sample_thread):
        """Test successful response generation"""
        mock_response = {
            "output": {"message": {"content": [{"text": "SCORE: 8\nJUSTIFICATION: Good"}]}},
        }

        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", True):
            with patch.object(backend, "_get_bedrock_client") as mock_client_method:
                mock_client = MagicMock()
                mock_client.converse.return_value = mock_response
                mock_client_method.return_value = mock_client

                response = await backend.generate_response(sample_thread)
                assert response == "SCORE: 8\nJUSTIFICATION: Good"
                mock_client.converse.assert_called_once()

    @pytest.mark.asyncio()
    async def test_generate_response_not_available(self, backend, sample_thread):
        """Test response generation when boto3 is not available"""
        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", False):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await backend.generate_response(sample_thread)
            assert "boto3 is not installed" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_generate_response_api_error(self, backend, sample_thread):
        """Test handling of Bedrock API errors"""
        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", True):
            with patch.object(backend, "_get_bedrock_client") as mock_client_method:
                from botocore.exceptions import ClientError

                mock_client = MagicMock()
                mock_client.converse.side_effect = ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "Invalid model"}},
                    "converse",
                )
                mock_client_method.return_value = mock_client

                with pytest.raises(LLMAPIError) as exc_info:
                    await backend.generate_response(sample_thread)
                assert exc_info.value.backend == "bedrock"

    def test_client_caching(self, backend):
        """Test that Bedrock client is cached properly"""
        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", True):
            with patch("decision_matrix_mcp.backends.bedrock.boto3.Session") as mock_session:
                mock_client = MagicMock()
                mock_session.return_value.client.return_value = mock_client

                # First call should create client
                client1 = backend._get_bedrock_client()
                assert client1 is mock_client

                # Second call should return cached client
                client2 = backend._get_bedrock_client()
                assert client2 is client1

                # Session should only be called once
                assert mock_session.call_count == 1

    def test_cleanup(self, backend):
        """Test cleanup method"""
        with patch("decision_matrix_mcp.backends.bedrock.BOTO3_AVAILABLE", True):
            with patch("decision_matrix_mcp.backends.bedrock.boto3.Session"):
                # Set up a client
                backend._get_bedrock_client()
                assert backend._bedrock_client is not None

                # Cleanup should clear the client
                backend.cleanup()
                assert backend._bedrock_client is None


class TestLiteLLMBackend:
    """Test LiteLLMBackend implementation"""

    @pytest.fixture()
    def backend(self):
        """Create a LiteLLMBackend instance"""
        return LiteLLMBackend()

    @pytest.fixture()
    def sample_thread(self):
        """Create a sample criterion thread"""
        criterion = Criterion(
            name="Performance",
            description="Evaluate performance",
            weight=2.0,
            model_backend=ModelBackend.LITELLM,
        )
        thread = CriterionThread(id="test-thread", criterion=criterion)
        thread.add_message("user", "Test message")
        return thread

    def test_is_available_with_litellm(self, backend):
        """Test availability checking when litellm is available"""
        with patch("decision_matrix_mcp.backends.litellm.LITELLM_AVAILABLE", True):
            assert backend.is_available() is True

    def test_is_available_without_litellm(self, backend):
        """Test availability checking when litellm is not available"""
        with patch("decision_matrix_mcp.backends.litellm.LITELLM_AVAILABLE", False):
            assert backend.is_available() is False

    @pytest.mark.asyncio()
    async def test_generate_response_success(self, backend, sample_thread):
        """Test successful response generation"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="SCORE: 7\nJUSTIFICATION: Good")),
        ]

        with (
            patch("decision_matrix_mcp.backends.litellm.LITELLM_AVAILABLE", True),
            patch(
                "decision_matrix_mcp.backends.litellm.litellm.acompletion",
                return_value=mock_response,
            ) as mock_litellm,
        ):
            response = await backend.generate_response(sample_thread)
            assert response == "SCORE: 7\nJUSTIFICATION: Good"
            mock_litellm.assert_called_once()

    @pytest.mark.asyncio()
    async def test_generate_response_not_available(self, backend, sample_thread):
        """Test response generation when litellm is not available"""
        with patch("decision_matrix_mcp.backends.litellm.LITELLM_AVAILABLE", False):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await backend.generate_response(sample_thread)
            assert "litellm is not installed" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_generate_response_api_error(self, backend, sample_thread):
        """Test handling of LiteLLM API errors"""
        with (
            patch("decision_matrix_mcp.backends.litellm.LITELLM_AVAILABLE", True),
            patch(
                "decision_matrix_mcp.backends.litellm.litellm.acompletion",
                side_effect=Exception("API key invalid"),
            ),
        ):
            with pytest.raises(LLMAPIError) as exc_info:
                await backend.generate_response(sample_thread)
            assert exc_info.value.backend == "litellm"
            assert "API authentication failed" in exc_info.value.user_message


class TestOllamaBackend:
    """Test OllamaBackend implementation"""

    @pytest.fixture()
    def backend(self):
        """Create an OllamaBackend instance"""
        return OllamaBackend()

    @pytest.fixture()
    def sample_thread(self):
        """Create a sample criterion thread"""
        criterion = Criterion(
            name="Performance",
            description="Evaluate performance",
            weight=2.0,
            model_backend=ModelBackend.OLLAMA,
        )
        thread = CriterionThread(id="test-thread", criterion=criterion)
        thread.add_message("user", "Test message")
        return thread

    def test_is_available_with_httpx(self, backend):
        """Test availability checking when httpx is available"""
        with patch("decision_matrix_mcp.backends.ollama.HTTPX_AVAILABLE", True):
            assert backend.is_available() is True

    def test_is_available_without_httpx(self, backend):
        """Test availability checking when httpx is not available"""
        with patch("decision_matrix_mcp.backends.ollama.HTTPX_AVAILABLE", False):
            assert backend.is_available() is False

    @pytest.mark.asyncio()
    async def test_generate_response_success(self, backend, sample_thread):
        """Test successful response generation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "SCORE: 6\nJUSTIFICATION: Average"},
        }

        with patch("decision_matrix_mcp.backends.ollama.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                response = await backend.generate_response(sample_thread)
                assert response == "SCORE: 6\nJUSTIFICATION: Average"
                mock_client.post.assert_called_once()

    @pytest.mark.asyncio()
    async def test_generate_response_not_available(self, backend, sample_thread):
        """Test response generation when httpx is not available"""
        with patch("decision_matrix_mcp.backends.ollama.HTTPX_AVAILABLE", False):
            with pytest.raises(LLMConfigurationError) as exc_info:
                await backend.generate_response(sample_thread)
            assert "httpx is not installed" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_generate_response_api_error(self, backend, sample_thread):
        """Test handling of Ollama API errors"""
        with patch("decision_matrix_mcp.backends.ollama.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.side_effect = Exception("Connection refused")
                mock_client_class.return_value = mock_client

                with pytest.raises(LLMAPIError) as exc_info:
                    await backend.generate_response(sample_thread)
                assert exc_info.value.backend == "ollama"
