"""Additional tests for validation_decorators.py to achieve 100% coverage."""
import pytest
from unittest.mock import Mock, patch
from decision_matrix_mcp import start_decision_analysis, add_criterion


class TestValidationDecoratorsAdditional:
    """Test validation decorator edge cases"""
    
    @pytest.mark.asyncio
    async def test_validation_with_ctx_parameter(self):
        """Test lines 30, 32: Decorated functions with ctx parameter"""
        # MCP framework passes a ctx parameter to tool functions
        ctx = Mock()
        ctx.logger = Mock()
        
        # Test with None request (should trigger validation error)
        result = await start_decision_analysis(None, ctx)
        assert "error" in result
        assert "Invalid request format" in result["error"]
        
        # Test with empty dict request
        result = await start_decision_analysis({}, ctx)
        assert "error" in result
        
        # Test with valid request
        with patch('decision_matrix_mcp.session_manager') as mock_sm:
            mock_session = Mock()
            mock_session.session_id = "test-session-123"
            mock_sm.create_session.return_value = mock_session
            
            result = await start_decision_analysis({
                "topic": "Test Decision",
                "options": ["Option A", "Option B", "Option C"]
            }, ctx)
            
            assert "session_id" in result
            assert result["session_id"] == "test-session-123"
    
    @pytest.mark.asyncio
    async def test_validation_with_missing_request(self):
        """Test validation when request parameter is missing"""
        ctx = Mock()
        
        # Test add_criterion with ctx but no request
        result = await add_criterion(None, ctx)
        assert "error" in result
        assert "Invalid request format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_validation_with_non_dict_request(self):
        """Test validation with non-dict request types"""
        ctx = Mock()
        
        # Test with string request
        result = await start_decision_analysis("not a dict", ctx)
        assert "error" in result
        assert "Invalid request format" in result["error"]
        
        # Test with list request
        result = await start_decision_analysis(["not", "a", "dict"], ctx)
        assert "error" in result
        assert "Invalid request format" in result["error"]
        
        # Test with number
        result = await start_decision_analysis(12345, ctx)
        assert "error" in result
        assert "Invalid request format" in result["error"]