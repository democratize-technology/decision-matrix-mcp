"""Tests for 100% coverage of orchestrator.py"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decision_matrix_mcp.exceptions import ConfigurationError, LLMBackendError
from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend, Option
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestOrchestratorCoverage:
    """Test missing coverage in orchestrator.py"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return DecisionOrchestrator()

    @pytest.fixture
    def sample_threads(self):
        """Create sample criterion threads"""
        criterion1 = Criterion(
            name="Cost",
            description="Evaluate cost effectiveness",
            weight=2.0,
            model_backend=ModelBackend.BEDROCK,
        )
        criterion2 = Criterion(
            name="Performance",
            description="Evaluate performance",
            weight=1.5,
            model_backend=ModelBackend.LITELLM,
        )

        threads = {
            "Cost": CriterionThread(id="thread1", criterion=criterion1),
            "Performance": CriterionThread(id="thread2", criterion=criterion2),
        }
        return threads

    @pytest.fixture
    def sample_options(self):
        """Create sample options"""
        return [
            Option(name="Option A", description="First option"),
            Option(name="Option B", description="Second option"),
        ]

    @pytest.mark.asyncio
    async def test_bedrock_client_creation_success(self, orchestrator):
        """Test successful Bedrock client creation"""
        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3:
                mock_client = MagicMock()
                mock_boto3.client.return_value = mock_client

                client = await orchestrator._get_or_create_bedrock_client()

                assert client == mock_client
                assert orchestrator._bedrock_client == mock_client
                mock_boto3.client.assert_called_once_with("bedrock-runtime")

    @pytest.mark.asyncio
    async def test_bedrock_client_creation_no_profile(self, orchestrator):
        """Test Bedrock client creation without AWS_PROFILE"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                await orchestrator._get_or_create_bedrock_client()
            assert "AWS_PROFILE environment variable not set" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bedrock_client_creation_import_error(self, orchestrator):
        """Test Bedrock client creation with import error"""
        with patch.dict(os.environ, {"AWS_PROFILE": "test"}):
            with patch("decision_matrix_mcp.orchestrator.boto3", None):
                with pytest.raises(ConfigurationError) as exc_info:
                    await orchestrator._get_or_create_bedrock_client()
                assert "boto3 is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_litellm_client_creation_success(self, orchestrator):
        """Test successful LiteLLM client creation"""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}):
            with patch("decision_matrix_mcp.orchestrator.litellm") as mock_litellm:
                client = await orchestrator._get_or_create_litellm_client()
                assert client == mock_litellm
                assert orchestrator._litellm_client == mock_litellm

    @pytest.mark.asyncio
    async def test_litellm_client_creation_no_key(self, orchestrator):
        """Test LiteLLM client creation without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                await orchestrator._get_or_create_litellm_client()
            assert "LITELLM_API_KEY environment variable not set" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_litellm_client_creation_import_error(self, orchestrator):
        """Test LiteLLM client creation with import error"""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "test"}):
            with patch("decision_matrix_mcp.orchestrator.litellm", None):
                with pytest.raises(ConfigurationError) as exc_info:
                    await orchestrator._get_or_create_litellm_client()
                assert "litellm is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ollama_client_creation_success(self, orchestrator):
        """Test successful Ollama client creation"""
        with patch("decision_matrix_mcp.orchestrator.ollama") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.AsyncClient.return_value = mock_client

            client = await orchestrator._get_or_create_ollama_client()

            assert client == mock_client
            assert orchestrator._ollama_client == mock_client
            mock_ollama.AsyncClient.assert_called_once_with(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )

    @pytest.mark.asyncio
    async def test_ollama_client_creation_import_error(self, orchestrator):
        """Test Ollama client creation with import error"""
        with patch("decision_matrix_mcp.orchestrator.ollama", None):
            with pytest.raises(ConfigurationError) as exc_info:
                await orchestrator._get_or_create_ollama_client()
            assert "ollama is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bedrock_backend_error_handling(self, orchestrator, sample_threads):
        """Test Bedrock backend error handling"""
        thread = sample_threads["Cost"]
        option = Option(name="Test Option")

        with patch.object(
            orchestrator, "_get_or_create_bedrock_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Simulate Bedrock API error
            mock_client.invoke_model.side_effect = Exception("Bedrock API error")

            score, justification = await orchestrator._evaluate_with_bedrock(
                thread, option
            )

            assert score is None
            assert "Error: Failed to get response from Bedrock" in justification

    @pytest.mark.asyncio
    async def test_litellm_backend_error_handling(self, orchestrator, sample_threads):
        """Test LiteLLM backend error handling"""
        thread = sample_threads["Performance"]
        option = Option(name="Test Option")

        with patch.object(
            orchestrator, "_get_or_create_litellm_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Simulate LiteLLM API error
            mock_client.acompletion.side_effect = Exception("LiteLLM API error")

            score, justification = await orchestrator._evaluate_with_litellm(
                thread, option
            )

            assert score is None
            assert "Error: Failed to get response from LiteLLM" in justification

    @pytest.mark.asyncio
    async def test_ollama_backend_json_parsing_error(self, orchestrator):
        """Test Ollama backend with JSON parsing error"""
        criterion = Criterion(
            name="Test",
            description="Test criterion",
            model_backend=ModelBackend.OLLAMA,
        )
        thread = CriterionThread(id="test", criterion=criterion)
        option = Option(name="Test Option")

        with patch.object(
            orchestrator, "_get_or_create_ollama_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            # Return invalid JSON
            mock_response = MagicMock()
            mock_response.message.content = "Invalid JSON response"
            mock_client.chat.return_value = mock_response

            score, justification = await orchestrator._evaluate_with_ollama(
                thread, option
            )

            assert score is None
            assert "Error parsing Ollama response" in justification

    @pytest.mark.asyncio
    async def test_evaluate_single_backend_switch(self, orchestrator):
        """Test backend switching in _evaluate_single"""
        option = Option(name="Test Option")

        # Test Ollama backend
        criterion_ollama = Criterion(
            name="Ollama Test",
            description="Test",
            model_backend=ModelBackend.OLLAMA,
        )
        thread_ollama = CriterionThread(id="ollama", criterion=criterion_ollama)

        with patch.object(
            orchestrator, "_evaluate_with_ollama"
        ) as mock_ollama:
            mock_ollama.return_value = (8.0, "Good")
            
            score, justification = await orchestrator._evaluate_single(
                thread_ollama, option
            )
            
            assert score == 8.0
            assert justification == "Good"
            mock_ollama.assert_called_once_with(thread_ollama, option)

    @pytest.mark.asyncio
    async def test_circuit_breaker_trip(self, orchestrator):
        """Test circuit breaker functionality"""
        criterion = Criterion(
            name="Test",
            description="Test",
            model_backend=ModelBackend.BEDROCK,
        )
        thread = CriterionThread(id="test", criterion=criterion)
        option = Option(name="Test")

        # Force multiple failures to trip circuit breaker
        with patch.object(
            orchestrator, "_get_or_create_bedrock_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.invoke_model.side_effect = Exception("API Error")

            # Fail multiple times
            for i in range(6):  # More than failure threshold
                score, _ = await orchestrator._evaluate_with_bedrock(thread, option)
                assert score is None

            # Next call should fail immediately (circuit open)
            score, justification = await orchestrator._evaluate_with_bedrock(
                thread, option
            )
            assert score is None
            assert "Circuit breaker open" in justification

    @pytest.mark.asyncio
    async def test_response_parsing_edge_cases(self, orchestrator):
        """Test response parser edge cases"""
        # Test with NO_RESPONSE
        parsed = orchestrator._response_parser._extract_score_and_justification(
            "SCORE: [NO_RESPONSE]\nJUSTIFICATION: Not applicable"
        )
        assert parsed == (None, "Not applicable")

        # Test with decimal score
        parsed = orchestrator._response_parser._extract_score_and_justification(
            "SCORE: 7.5\nJUSTIFICATION: Good"
        )
        assert parsed == (7.5, "Good")

        # Test with out of range score
        parsed = orchestrator._response_parser._extract_score_and_justification(
            "SCORE: 15\nJUSTIFICATION: Too high"
        )
        assert parsed == (None, "Score out of range (1-10): 15.0")

        # Test with missing justification
        parsed = orchestrator._response_parser._extract_score_and_justification(
            "SCORE: 8"
        )
        assert parsed == (8.0, "No justification provided")

    @pytest.mark.asyncio
    async def test_evaluate_options_empty_inputs(self, orchestrator):
        """Test evaluate_options with empty inputs"""
        # Empty threads
        results = await orchestrator.evaluate_options_across_criteria({}, [])
        assert results == {}

        # Empty options
        threads = {"Test": CriterionThread(id="test", criterion=Criterion("Test", "Test"))}
        results = await orchestrator.evaluate_options_across_criteria(threads, [])
        assert results == {"Test": {}}

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_partial_failure(
        self, orchestrator, sample_threads, sample_options
    ):
        """Test concurrent evaluation with partial failures"""
        # Mock one success and one failure
        async def mock_evaluate(thread, option):
            if thread.criterion.name == "Cost":
                return (8.0, "Good cost")
            else:
                raise Exception("Evaluation failed")

        with patch.object(orchestrator, "_evaluate_single", side_effect=mock_evaluate):
            results = await orchestrator.evaluate_options_across_criteria(
                sample_threads, sample_options
            )

            # Cost should succeed for both options
            assert results["Cost"]["Option A"] == (8.0, "Good cost")
            assert results["Cost"]["Option B"] == (8.0, "Good cost")

            # Performance should fail for both options
            assert results["Performance"]["Option A"][0] is None
            assert "Error evaluating" in results["Performance"]["Option A"][1]
            assert results["Performance"]["Option B"][0] is None
            assert "Error evaluating" in results["Performance"]["Option B"][1]

    def test_json_response_parsing_variations(self, orchestrator):
        """Test JSON response parsing with various formats"""
        parser = orchestrator._response_parser

        # Valid JSON formats
        test_cases = [
            ('{"score": 8, "justification": "Good"}', (8.0, "Good")),
            ('{"SCORE": 7.5, "JUSTIFICATION": "OK"}', (7.5, "OK")),
            ('{"score": "9", "justification": "Excellent"}', (9.0, "Excellent")),
            ('{"score": "[NO_RESPONSE]", "justification": "N/A"}', (None, "N/A")),
            ('{"score": null, "justification": "None"}', (None, "None")),
        ]

        for json_str, expected in test_cases:
            result = parser._parse_json_response(json_str)
            assert result == expected

        # Invalid JSON
        result = parser._parse_json_response("Not JSON")
        assert result == (None, "Invalid JSON response format")

        # Missing fields
        result = parser._parse_json_response('{"score": 8}')
        assert result == (8.0, "No justification provided")

        result = parser._parse_json_response('{"justification": "Good"}')
        assert result == (None, "Good")

    @pytest.mark.asyncio
    async def test_backend_specific_model_names(self, orchestrator):
        """Test using specific model names for each backend"""
        option = Option(name="Test")

        # Bedrock with specific model
        criterion_bedrock = Criterion(
            name="Test",
            description="Test",
            model_backend=ModelBackend.BEDROCK,
            model_name="anthropic.claude-3-sonnet",
        )
        thread_bedrock = CriterionThread(id="bedrock", criterion=criterion_bedrock)

        with patch.object(
            orchestrator, "_get_or_create_bedrock_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_response = {
                "body": MagicMock(
                    read=MagicMock(
                        return_value=b'{"content": "SCORE: 8\\nJUSTIFICATION: Good"}'
                    )
                )
            }
            mock_client.invoke_model.return_value = mock_response

            score, justification = await orchestrator._evaluate_with_bedrock(
                thread_bedrock, option
            )

            assert score == 8.0
            assert justification == "Good"
            # Verify model ID was used
            mock_client.invoke_model.assert_called_once()
            call_args = mock_client.invoke_model.call_args[1]
            assert call_args["modelId"] == "anthropic.claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_retry_logic_success_on_second_attempt(self, orchestrator):
        """Test retry logic succeeding on second attempt"""
        criterion = Criterion(
            name="Test",
            description="Test",
            model_backend=ModelBackend.BEDROCK,
        )
        thread = CriterionThread(id="test", criterion=criterion)
        option = Option(name="Test")

        call_count = 0

        async def mock_evaluate(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return (8.0, "Success on retry")

        with patch.object(
            orchestrator, "_evaluate_with_bedrock", side_effect=mock_evaluate
        ):
            score, justification = await orchestrator._evaluate_single(thread, option)

            assert score == 8.0
            assert justification == "Success on retry"
            assert call_count == 2  # First attempt failed, second succeeded