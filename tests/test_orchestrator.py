"""Tests for the DecisionOrchestrator"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decision_matrix_mcp.exceptions import (
    ConfigurationError,
    LLMAPIError,
    LLMBackendError,
    LLMConfigurationError,
)
from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend, Option
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestDecisionOrchestrator:
    """Test the DecisionOrchestrator class"""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance"""
        return DecisionOrchestrator(max_retries=2, retry_delay=0.1)

    @pytest.fixture
    def sample_criterion(self):
        """Create a sample criterion"""
        return Criterion(
            name="Performance",
            description="Evaluate performance",
            weight=2.0,
            model_backend=ModelBackend.BEDROCK,
        )

    @pytest.fixture
    def sample_thread(self, sample_criterion):
        """Create a sample criterion thread"""
        return CriterionThread(id="test-thread", criterion=sample_criterion)

    @pytest.fixture
    def sample_options(self):
        """Create sample options"""
        return [
            Option(name="Option A", description="First option"),
            Option(name="Option B", description="Second option"),
        ]

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = DecisionOrchestrator(max_retries=5, retry_delay=2.0)
        assert orchestrator.max_retries == 5
        assert orchestrator.retry_delay == 2.0
        assert len(orchestrator.backends) == 3
        assert ModelBackend.BEDROCK in orchestrator.backends
        assert ModelBackend.LITELLM in orchestrator.backends
        assert ModelBackend.OLLAMA in orchestrator.backends

    def test_parse_evaluation_response_valid_score(self, orchestrator):
        """Test parsing valid evaluation response"""
        response = """SCORE: 8
JUSTIFICATION: This option performs well due to its efficiency."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 8.0
        assert justification == "This option performs well due to its efficiency."

    def test_parse_evaluation_response_decimal_score(self, orchestrator):
        """Test parsing decimal score"""
        response = """SCORE: 7.5
JUSTIFICATION: Good but not perfect."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 7.5
        assert justification == "Good but not perfect."

    def test_parse_evaluation_response_out_of_range_scores(self, orchestrator):
        """Test clamping scores to 1-10 range"""
        # Test score above 10
        response = """SCORE: 15
JUSTIFICATION: Exceeds expectations."""
        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 10.0

        # Test score below 1
        response = """SCORE: 0.5
JUSTIFICATION: Below minimum."""
        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 1.0

    def test_parse_evaluation_response_no_response(self, orchestrator):
        """Test parsing NO_RESPONSE abstention"""
        response = """SCORE: [NO_RESPONSE]
JUSTIFICATION: This criterion doesn't apply to this option."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score is None
        assert justification == "This criterion doesn't apply to this option."

    def test_parse_evaluation_response_no_response_variations(self, orchestrator):
        """Test parsing different NO_RESPONSE variations"""
        variations = [
            "SCORE: NO_RESPONSE",
            "SCORE: [NO_RESPONSE]",
            "Score: NO_RESPONSE - not applicable",
            "SCORE: I'm giving NO_RESPONSE here",
        ]

        for response in variations:
            response_full = f"{response}\nJUSTIFICATION: Not applicable"
            score, _ = orchestrator._parse_evaluation_response(response_full)
            assert score is None

    def test_parse_evaluation_response_missing_score(self, orchestrator):
        """Test parsing response with missing score"""
        response = """JUSTIFICATION: This is a good option."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score is None
        assert justification == "This is a good option."

    def test_parse_evaluation_response_missing_justification(self, orchestrator):
        """Test parsing response with missing justification"""
        response = """SCORE: 7"""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 7.0
        assert justification == "No justification provided"

    def test_parse_evaluation_response_multiline_justification(self, orchestrator):
        """Test parsing multiline justification"""
        response = """SCORE: 9
JUSTIFICATION: This option is excellent for several reasons:
- High performance
- Cost effective
- Easy to implement"""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 9.0
        assert "High performance" in justification
        assert "Cost effective" in justification
        assert "Easy to implement" in justification

    def test_parse_evaluation_response_case_insensitive(self, orchestrator):
        """Test case insensitive parsing"""
        response = """score: 6
justification: Average performance."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 6.0
        assert justification == "Average performance."

    def test_parse_evaluation_response_parse_error(self, orchestrator):
        """Test handling parse errors"""
        response = """SCORE: not-a-number
JUSTIFICATION: Invalid score."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score is None

    @pytest.mark.asyncio
    async def test_evaluate_single_option_success(
        self, orchestrator, sample_thread, sample_options
    ):
        """Test successful single option evaluation"""
        mock_response = """SCORE: 8
JUSTIFICATION: Strong performance characteristics."""

        with patch.object(orchestrator, "_get_thread_response", return_value=mock_response):
            score, justification = await orchestrator._evaluate_single_option(
                sample_thread, sample_options[0]
            )

            assert score == 8.0
            assert justification == "Strong performance characteristics."
            assert len(sample_thread.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_evaluate_single_option_abstention(
        self, orchestrator, sample_thread, sample_options
    ):
        """Test option evaluation with abstention"""
        mock_response = """SCORE: [NO_RESPONSE]
JUSTIFICATION: Performance criteria not applicable to this option."""

        with patch.object(orchestrator, "_get_thread_response", return_value=mock_response):
            score, justification = await orchestrator._evaluate_single_option(
                sample_thread, sample_options[0]
            )

            assert score is None
            assert "not applicable" in justification

    @pytest.mark.asyncio
    async def test_evaluate_single_option_llm_error(
        self, orchestrator, sample_thread, sample_options
    ):
        """Test handling LLM backend errors"""
        error = LLMBackendError(
            backend="test",
            message="API rate limit exceeded",
            user_message="Rate limit reached. Try again later.",
        )

        with patch.object(orchestrator, "_get_thread_response", side_effect=error):
            score, justification = await orchestrator._evaluate_single_option(
                sample_thread, sample_options[0]
            )

            assert score is None
            assert justification == "Rate limit reached. Try again later."

    @pytest.mark.asyncio
    async def test_evaluate_single_option_unexpected_error(
        self, orchestrator, sample_thread, sample_options
    ):
        """Test handling unexpected errors"""
        with patch.object(
            orchestrator, "_get_thread_response", side_effect=Exception("Unknown error")
        ):
            score, justification = await orchestrator._evaluate_single_option(
                sample_thread, sample_options[0]
            )

            assert score is None
            assert justification == "Evaluation failed due to an unexpected error"

    @pytest.mark.asyncio
    async def test_evaluate_options_across_criteria(self, orchestrator, sample_options):
        """Test evaluating multiple options across multiple criteria"""
        # Create two criteria threads
        criterion1 = Criterion(name="Performance", description="Evaluate performance", weight=2.0)
        criterion2 = Criterion(name="Cost", description="Evaluate cost", weight=1.5)

        threads = {
            "Performance": CriterionThread(id="thread-1", criterion=criterion1),
            "Cost": CriterionThread(id="thread-2", criterion=criterion2),
        }

        # Mock responses
        mock_responses = {
            ("Performance", "Option A"): (8.0, "Good performance"),
            ("Performance", "Option B"): (6.0, "Average performance"),
            ("Cost", "Option A"): (7.0, "Reasonable cost"),
            ("Cost", "Option B"): (9.0, "Very cost effective"),
        }

        async def mock_evaluate(thread, option):
            key = (thread.criterion.name, option.name)
            return mock_responses[key]

        with patch.object(orchestrator, "_evaluate_single_option", side_effect=mock_evaluate):
            results = await orchestrator.evaluate_options_across_criteria(threads, sample_options)

            assert len(results) == 2
            assert results["Performance"]["Option A"] == (8.0, "Good performance")
            assert results["Performance"]["Option B"] == (6.0, "Average performance")
            assert results["Cost"]["Option A"] == (7.0, "Reasonable cost")
            assert results["Cost"]["Option B"] == (9.0, "Very cost effective")

    @pytest.mark.asyncio
    async def test_evaluate_options_with_exceptions(self, orchestrator, sample_options):
        """Test handling exceptions during parallel evaluation"""
        criterion = Criterion(name="Performance", description="Evaluate performance", weight=2.0)
        threads = {"Performance": CriterionThread(id="thread-1", criterion=criterion)}

        # Make first evaluation fail
        side_effects = [Exception("Evaluation failed"), (7.0, "Good option")]

        with patch.object(orchestrator, "_evaluate_single_option", side_effect=side_effects):
            results = await orchestrator.evaluate_options_across_criteria(threads, sample_options)

            assert results["Performance"]["Option A"] == (None, "Error: Evaluation failed")
            assert results["Performance"]["Option B"] == (7.0, "Good option")

    @pytest.mark.asyncio
    async def test_get_thread_response_unknown_backend(self, orchestrator):
        """Test error handling for unknown backend"""
        criterion = Criterion(
            name="Test",
            description="Test criterion",
            weight=1.0,
            model_backend="unknown_backend",  # Invalid backend
        )
        thread = CriterionThread(id="test-thread", criterion=criterion)

        with pytest.raises(ConfigurationError) as exc_info:
            await orchestrator._get_thread_response(thread)

        assert "Unknown model backend" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_thread_response_retry_logic(self, orchestrator, sample_thread):
        """Test retry logic with transient errors"""
        # Mock backend function that fails once then succeeds
        call_count = 0

        async def mock_backend(thread):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary network error")
            return "Success response"

        orchestrator.backends[ModelBackend.BEDROCK] = mock_backend

        # Mock asyncio.sleep to avoid real delays in tests
        with patch("asyncio.sleep"):
            response = await orchestrator._get_thread_response(sample_thread)
            assert response == "Success response"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_thread_response_non_retryable_errors(self, orchestrator, sample_thread):
        """Test that non-retryable errors fail immediately"""
        non_retryable_errors = [
            "Invalid API_KEY",
            "Credentials not found",
            "Model not found",
            "Unauthorized access",
            "Forbidden resource",
        ]

        for error_msg in non_retryable_errors:

            async def mock_backend(thread, msg=error_msg):
                raise Exception(msg)

            orchestrator.backends[ModelBackend.BEDROCK] = mock_backend

            with pytest.raises(Exception) as exc_info:
                await orchestrator._get_thread_response(sample_thread)

            assert error_msg in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_thread_response_max_retries_exceeded(self, orchestrator, sample_thread):
        """Test max retries exceeded"""

        async def mock_backend(thread):
            raise Exception("Persistent error")

        orchestrator.backends[ModelBackend.BEDROCK] = mock_backend
        orchestrator.max_retries = 2

        with patch("asyncio.sleep"):
            with pytest.raises(Exception) as exc_info:
                await orchestrator._get_thread_response(sample_thread)

            assert "Persistent error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_bedrock_success(self, orchestrator, sample_thread):
        """Test successful Bedrock API call"""
        mock_response = {
            "body": MagicMock(
                read=lambda: b'{"content": [{"text": "SCORE: 8\\nJUSTIFICATION: Good"}]}'
            )
        }

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_session.return_value.client.return_value = mock_client

            response = await orchestrator._call_bedrock(sample_thread)
            assert response == "SCORE: 8\nJUSTIFICATION: Good"
            
            # Verify request body
            call_args = mock_client.invoke_model.call_args
            request_body = json.loads(call_args.kwargs["body"])
            assert request_body["temperature"] == sample_thread.criterion.temperature
            assert request_body["max_tokens"] == sample_thread.criterion.max_tokens
            # Seed should not be in request if None
            assert "seed" not in request_body

    @pytest.mark.asyncio
    async def test_call_bedrock_with_seed(self, orchestrator):
        """Test Bedrock API call with custom seed"""
        # Create criterion with seed
        criterion = Criterion(
            name="Performance",
            description="Test",
            weight=1.0,
            temperature=0.5,
            seed=42,
            max_tokens=512
        )
        thread = CriterionThread(id="test", criterion=criterion)
        
        mock_response = {
            "body": MagicMock(
                read=lambda: b'{"content": [{"text": "SCORE: 8\\nJUSTIFICATION: Good"}]}'
            )
        }
        
        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_session.return_value.client.return_value = mock_client
            
            await orchestrator._call_bedrock(thread)
            
            # Verify seed is included
            call_args = mock_client.invoke_model.call_args
            request_body = json.loads(call_args.kwargs["body"])
            assert request_body["seed"] == 42
            assert request_body["temperature"] == 0.5
            assert request_body["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_call_bedrock_import_error(self, orchestrator, sample_thread):
        """Test Bedrock call when boto3 not available"""
        # Temporarily patch the BOTO3_AVAILABLE flag
        import sys
        orch_module = sys.modules['decision_matrix_mcp.orchestrator']
        original_value = orch_module.BOTO3_AVAILABLE
        try:
            orch_module.BOTO3_AVAILABLE = False
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._call_bedrock(sample_thread)

            assert "boto3 is not installed" in str(exc_info.value)
        finally:
            orch_module.BOTO3_AVAILABLE = original_value

    @pytest.mark.asyncio
    async def test_call_bedrock_api_error(self, orchestrator, sample_thread):
        """Test Bedrock API errors"""
        with patch("boto3.Session") as mock_session:
            from botocore.exceptions import ClientError

            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Invalid model"}},
                "invoke_model",
            )
            mock_session.return_value.client.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_bedrock(sample_thread)

            assert "Bedrock API call failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_bedrock_invalid_response(self, orchestrator, sample_thread):
        """Test handling invalid Bedrock response format"""
        mock_response = {"body": MagicMock(read=lambda: b'{"invalid": "format"}')}

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_session.return_value.client.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_bedrock(sample_thread)

            assert "Invalid response format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_litellm_success(self, orchestrator, sample_thread):
        """Test successful LiteLLM API call"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="SCORE: 7\nJUSTIFICATION: Good"))
        ]

        with patch("litellm.acompletion", return_value=mock_response) as mock_litellm:
            response = await orchestrator._call_litellm(sample_thread)
            assert response == "SCORE: 7\nJUSTIFICATION: Good"

            # Verify call parameters
            mock_litellm.assert_called_once()
            call_args = mock_litellm.call_args
            assert call_args.kwargs["temperature"] == sample_thread.criterion.temperature
            assert call_args.kwargs["max_tokens"] == sample_thread.criterion.max_tokens
            assert call_args.kwargs["seed"] == sample_thread.criterion.seed

    @pytest.mark.asyncio
    async def test_call_litellm_import_error(self, orchestrator, sample_thread):
        """Test LiteLLM call when litellm not available"""
        # Temporarily patch the LITELLM_AVAILABLE flag
        import sys
        orch_module = sys.modules['decision_matrix_mcp.orchestrator']
        original_value = orch_module.LITELLM_AVAILABLE
        try:
            orch_module.LITELLM_AVAILABLE = False
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._call_litellm(sample_thread)

            assert "litellm is not installed" in str(exc_info.value)
        finally:
            orch_module.LITELLM_AVAILABLE = original_value

    @pytest.mark.asyncio
    async def test_call_litellm_api_error(self, orchestrator, sample_thread):
        """Test LiteLLM API errors"""
        with patch("litellm.acompletion", side_effect=Exception("API key invalid")):
            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_litellm(sample_thread)

            assert "LiteLLM API call failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_ollama_success(self, orchestrator, sample_thread):
        """Test successful Ollama API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "SCORE: 6\nJUSTIFICATION: Average"}
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await orchestrator._call_ollama(sample_thread)
            assert response == "SCORE: 6\nJUSTIFICATION: Average"
            
            # Verify request parameters
            call_args = mock_client.post.call_args
            request_json = call_args.kwargs["json"]
            assert request_json["options"]["temperature"] == sample_thread.criterion.temperature
            # Seed should not be in options if None
            assert "seed" not in request_json["options"]

    @pytest.mark.asyncio
    async def test_call_ollama_with_seed(self, orchestrator):
        """Test Ollama API call with custom seed"""
        # Create criterion with seed
        criterion = Criterion(
            name="Test",
            description="Test",
            weight=1.0,
            temperature=0.7,
            seed=12345
        )
        thread = CriterionThread(id="test", criterion=criterion)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "SCORE: 6\nJUSTIFICATION: Test"}
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            await orchestrator._call_ollama(thread)
            
            # Verify seed is included
            call_args = mock_client.post.call_args
            request_json = call_args.kwargs["json"]
            assert request_json["options"]["temperature"] == 0.7
            assert request_json["options"]["seed"] == 12345

    @pytest.mark.asyncio
    async def test_call_ollama_custom_host(self, orchestrator, sample_thread):
        """Test Ollama with custom host from environment"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "SCORE: 5\nJUSTIFICATION: OK"}}

        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:11434"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                await orchestrator._call_ollama(sample_thread)

                # Verify custom host was used
                call_args = mock_client.post.call_args
                assert call_args.args[0] == "http://custom:11434/api/chat"

    @pytest.mark.asyncio
    async def test_call_ollama_import_error(self, orchestrator, sample_thread):
        """Test Ollama call when httpx not available"""
        # Temporarily patch the HTTPX_AVAILABLE flag
        import sys
        orch_module = sys.modules['decision_matrix_mcp.orchestrator']
        original_value = orch_module.HTTPX_AVAILABLE
        try:
            orch_module.HTTPX_AVAILABLE = False
            with pytest.raises(LLMConfigurationError) as exc_info:
                await orchestrator._call_ollama(sample_thread)

            assert "httpx is not installed" in str(exc_info.value)
        finally:
            orch_module.HTTPX_AVAILABLE = original_value

    @pytest.mark.asyncio
    async def test_call_ollama_api_error(self, orchestrator, sample_thread):
        """Test Ollama API errors"""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_ollama(sample_thread)

            assert exc_info.value.backend == "ollama"
            assert exc_info.value.user_message == "Model not available in Ollama: llama2"

    @pytest.mark.asyncio
    async def test_call_ollama_connection_error(self, orchestrator, sample_thread):
        """Test Ollama connection errors"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMAPIError) as exc_info:
                await orchestrator._call_ollama(sample_thread)

            assert "Ollama call failed" in str(exc_info.value)

    def test_parse_evaluation_response_edge_cases(self, orchestrator):
        """Test edge cases in response parsing"""
        # Empty response
        score, justification = orchestrator._parse_evaluation_response("")
        assert score is None
        assert "Could not parse evaluation from response" in justification

        # Only whitespace
        score, justification = orchestrator._parse_evaluation_response("   \n   \t   ")
        assert score is None
        assert "Could not parse evaluation from response" in justification

        # Score with extra text
        response = "SCORE: I think 8 would be fair\nJUSTIFICATION: Good option"
        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 8.0

        # Multiple scores (should take first)
        response = "SCORE: 7\nSCORE: 8\nJUSTIFICATION: Confusing"
        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 7.0
