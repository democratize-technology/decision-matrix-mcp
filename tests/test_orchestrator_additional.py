"""Additional tests for orchestrator.py to achieve 100% coverage."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from uuid import uuid4

from decision_matrix_mcp.orchestrator import DecisionOrchestrator
from decision_matrix_mcp.models import CriterionThread, Criterion


class TestOrchestratorAdditionalCoverage:
    """Additional test cases for full orchestrator coverage"""
    
    def test_parse_evaluation_response_partial_parse_exception(self):
        """Test lines 161-168: Exception during parsing with partial recovery"""
        orchestrator = DecisionOrchestrator()
        
        # Response that will cause initial parsing to fail
        response = "Invalid format but has JUSTIFICATION: This is a good justification"
        
        # Force _extract_score to raise an exception
        with patch.object(orchestrator, '_extract_score', side_effect=Exception("Score parse failed")):
            score, justification = orchestrator._parse_evaluation_response(response)
            
            # Should recover justification despite score parse failure
            assert score is None
            assert justification == "This is a good justification"
    
    def test_parse_evaluation_response_complete_parse_failure(self):
        """Test lines 167-168: Complete parse failure"""
        orchestrator = DecisionOrchestrator()
        
        response = "Completely unparseable response"
        
        # Force both methods to fail
        with patch.object(orchestrator, '_extract_score', side_effect=Exception("Score failed")):
            with patch.object(orchestrator, '_extract_justification', side_effect=Exception("Just failed")):
                score, justification = orchestrator._parse_evaluation_response(response)
                
                assert score is None
                assert justification == "Parse error: Unable to extract evaluation"
    
    def test_extract_score_value_error_in_conversion(self):
        """Test lines 192-193: ValueError during float conversion"""
        orchestrator = DecisionOrchestrator()
        
        # Test various invalid score formats
        invalid_scores = [
            "SCORE: not-a-number",
            "Score: 5.5.5",  # Multiple decimals
            "SCORE: infinity",
            "Rating: ten"
        ]
        
        for response in invalid_scores:
            score = orchestrator._extract_score(response)
            assert score is None
    
    @pytest.mark.asyncio
    async def test_call_bedrock_message_formatting(self):
        """Test line 295: Bedrock message content list formatting"""
        orchestrator = DecisionOrchestrator()
        
        # Create thread with messages
        criterion = Criterion(name="Quality", description="Quality assessment", weight=2.0)
        thread = CriterionThread(id=str(uuid4()), criterion=criterion)
        thread.add_message("user", "First message")
        thread.add_message("assistant", "Response")
        thread.add_message("user", "Second message")
        
        with patch('boto3.client') as mock_boto:
            mock_client = Mock()
            mock_response = {
                'output': {
                    'message': {
                        'content': [{'text': 'SCORE: 9\nJUSTIFICATION: Excellent'}]
                    }
                }
            }
            mock_client.converse.return_value = mock_response
            mock_boto.return_value = mock_client
            
            await orchestrator._call_bedrock(thread, "Evaluate")
            
            # Verify the message structure
            call_args = mock_client.converse.call_args
            messages = call_args.kwargs['messages']
            
            # Each message should have content as a list with type and text
            assert len(messages) == 3
            for msg in messages:
                assert 'content' in msg
                assert isinstance(msg['content'], list)
                assert msg['content'][0]['type'] == 'text'
                assert 'text' in msg['content'][0]
    
    @pytest.mark.asyncio
    async def test_call_litellm_edge_cases(self):
        """Test lines 338, 365: LiteLLM response edge cases"""
        orchestrator = DecisionOrchestrator()
        
        criterion = Criterion(name="Performance", description="Performance test", weight=1.5)
        thread = CriterionThread(id=str(uuid4()), criterion=criterion)
        
        # Test empty choices list
        with patch('litellm.acompletion') as mock_completion:
            mock_response = Mock()
            mock_response.choices = []
            mock_completion.return_value = mock_response
            
            result = await orchestrator._call_litellm(thread, "Test prompt")
            assert result == ""
        
        # Test None content in message
        with patch('litellm.acompletion') as mock_completion:
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = None
            mock_response.choices = [mock_choice]
            mock_completion.return_value = mock_response
            
            result = await orchestrator._call_litellm(thread, "Test prompt")
            assert result == ""
    
    @pytest.mark.asyncio
    async def test_call_ollama_malformed_responses(self):
        """Test lines 392-395, 412: Ollama malformed response handling"""
        orchestrator = DecisionOrchestrator()
        
        criterion = Criterion(name="Usability", description="Usability test", weight=1.0)
        thread = CriterionThread(id=str(uuid4()), criterion=criterion)
        
        # Test response without 'message' key
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {"error": "No message field"}
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            with pytest.raises(Exception, match="Unexpected Ollama response format"):
                await orchestrator._call_ollama(thread, "Test")
        
        # Test message without 'content' key
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "message": {"role": "assistant"}  # Missing content
            }
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await orchestrator._call_ollama(thread, "Test")
            assert result == ""
    
    @pytest.mark.asyncio
    async def test_get_thread_response_bedrock_retry_logic(self):
        """Test lines 437-438, 453: Bedrock retry on specific errors"""
        orchestrator = DecisionOrchestrator()
        
        criterion = Criterion(
            name="Reliability", 
            description="Reliability test", 
            weight=1.0,
            model_backend="bedrock"
        )
        thread = CriterionThread(id=str(uuid4()), criterion=criterion)
        
        # Create retryable error
        throttle_error = Exception("Request limit exceeded")
        throttle_error.__class__.__name__ = "ThrottlingException"
        
        # First two calls fail, third succeeds
        call_count = 0
        async def mock_bedrock_call(*args):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise throttle_error
            return "SCORE: 7\nJUSTIFICATION: Retry succeeded"
        
        with patch.object(orchestrator, '_call_bedrock', side_effect=mock_bedrock_call):
            result = await orchestrator._get_thread_response(
                thread, criterion, "TestOption", "Evaluate this"
            )
            
            assert result == "SCORE: 7\nJUSTIFICATION: Retry succeeded"
            assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_thread_response_non_retryable_errors(self):
        """Test line 477: Non-retryable errors are re-raised"""
        orchestrator = DecisionOrchestrator()
        
        criterion = Criterion(
            name="Security",
            description="Security test",
            weight=2.0,
            model_backend="litellm"
        )
        thread = CriterionThread(id=str(uuid4()), criterion=criterion)
        
        # Non-retryable error should be re-raised immediately
        auth_error = ValueError("Invalid API key")
        
        with patch.object(orchestrator, '_call_litellm', side_effect=auth_error):
            with pytest.raises(ValueError, match="Invalid API key"):
                await orchestrator._get_thread_response(
                    thread, criterion, "TestOption", "Evaluate"
                )