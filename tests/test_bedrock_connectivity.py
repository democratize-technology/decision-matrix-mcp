"""Tests for Bedrock connectivity testing functionality"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from decision_matrix_mcp.orchestrator import DecisionOrchestrator
from decision_matrix_mcp import test_aws_bedrock_connection as bedrock_connection_tool


class TestBedrockConnectivityOrchestrator:
    """Test the orchestrator's Bedrock connectivity test method"""

    @pytest.fixture
    def orchestrator(self):
        return DecisionOrchestrator()

    @pytest.mark.asyncio
    async def test_bedrock_connection_success(self, orchestrator):
        """Test successful Bedrock connection"""
        mock_response = {"output": {"message": {"content": [{"text": "Hello!"}]}}}

        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
            patch.dict("os.environ", {"AWS_REGION": "us-west-2"}),
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.return_value = mock_response
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "ok"
            assert result["region"] == "us-west-2"
            assert result["model_tested"] == "anthropic.claude-3-haiku-20240307-v1:0"
            assert result["response_length"] == 6  # len("Hello!")
            assert result["message"] == "Bedrock connection successful"
            assert result["api_version"] == "converse"

            # Verify the request was made correctly
            mock_bedrock.converse.assert_called_once()
            call_args = mock_bedrock.converse.call_args
            assert call_args[1]["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
            assert call_args[1]["inferenceConfig"]["maxTokens"] == 10
            assert call_args[1]["inferenceConfig"]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_bedrock_connection_boto3_unavailable(self, orchestrator):
        """Test when boto3 is not available"""
        with patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", False):
            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert result["error"] == "boto3 not installed. Install with: pip install boto3"
            assert result["region"] == "N/A"

    @pytest.mark.asyncio
    async def test_bedrock_connection_access_denied(self, orchestrator):
        """Test access denied error"""
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "You don't have access to the model with the specified model ID.",
            }
        }
        mock_error = ClientError(error_response, "InvokeModel")

        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
            patch.dict("os.environ", {"AWS_REGION": "us-east-1"}),
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.side_effect = mock_error
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert result["region"] == "us-east-1"
            assert result["error_code"] == "AccessDeniedException"
            assert "access" in result["error"].lower()
            assert "Enable model access in AWS Console" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_bedrock_connection_region_error(self, orchestrator):
        """Test region not available error"""
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "The requested region is not supported",
            }
        }
        mock_error = ClientError(error_response, "InvokeModel")

        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.side_effect = mock_error
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "region" in result["error"].lower()
            assert "Try us-east-1 or us-west-2" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_bedrock_connection_credentials_error(self, orchestrator):
        """Test credentials error"""
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {"Code": "UnauthorizedOperation", "Message": "Unable to locate credentials"}
        }
        mock_error = ClientError(error_response, "InvokeModel")

        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.side_effect = mock_error
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "credentials" in result["error"].lower()
            assert "Configure AWS credentials" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_bedrock_connection_throttling_error(self, orchestrator):
        """Test throttling error"""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        mock_error = ClientError(error_response, "InvokeModel")

        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.side_effect = mock_error
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "throttling" in result["error"].lower()
            assert "Request rate limit exceeded" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_bedrock_connection_unexpected_error(self, orchestrator):
        """Test unexpected error"""
        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.side_effect = Exception("Unexpected error")
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "Unexpected error" in result["error"]
            assert "Check AWS credentials and region configuration" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_bedrock_connection_default_region(self, orchestrator):
        """Test default region is used when no environment variable set"""
        mock_response = {"output": {"message": {"content": [{"text": "Hi"}]}}}

        with (
            patch("decision_matrix_mcp.orchestrator.BOTO3_AVAILABLE", True),
            patch("decision_matrix_mcp.orchestrator.boto3") as mock_boto3,
            patch.dict("os.environ", {}, clear=True),
        ):

            mock_session = MagicMock()
            mock_bedrock = MagicMock()
            mock_bedrock.converse.return_value = mock_response
            mock_session.client.return_value = mock_bedrock
            mock_boto3.Session.return_value = mock_session

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "ok"
            assert result["region"] == "us-east-1"  # Default region

    def test_get_bedrock_error_suggestion(self, orchestrator):
        """Test error suggestion logic"""
        # Access error
        suggestion = orchestrator._get_bedrock_error_suggestion("AccessDenied", "access denied")
        assert "Enable model access in AWS Console" in suggestion

        # Region error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "ValidationException", "region not available"
        )
        assert "Try us-east-1 or us-west-2" in suggestion

        # Credentials error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "UnauthorizedOperation", "credentials not found"
        )
        assert "Configure AWS credentials" in suggestion

        # Throttling error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "ThrottlingException", "throttling limit exceeded"
        )
        assert "Request rate limit exceeded" in suggestion

        # Unknown error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "UnknownError", "something went wrong"
        )
        assert "Check AWS Bedrock service status" in suggestion


class TestBedrockConnectivityMCPTool:
    """Test the MCP tool wrapper for Bedrock connectivity testing"""

    @pytest.mark.asyncio
    async def test_mcp_tool_success(self):
        """Test successful MCP tool call"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            # Mock the server components
            mock_components = MagicMock()
            mock_orchestrator = AsyncMock()
            mock_formatter = MagicMock()

            mock_orchestrator.test_bedrock_connection.return_value = {
                "status": "ok",
                "region": "us-west-2",
                "model_tested": "anthropic.claude-3-haiku-20240307-v1:0",
                "response_length": 6,
                "message": "Bedrock connection successful",
            }

            mock_components.orchestrator = mock_orchestrator
            mock_components.formatter = mock_formatter
            mock_get_components.return_value = mock_components

            result = await bedrock_connection_tool()

            assert result["status"] == "ok"
            assert result["region"] == "us-west-2"
            assert result["model_tested"] == "anthropic.claude-3-haiku-20240307-v1:0"
            assert "formatted_output" in result
            assert "✅ Bedrock Connection Test: SUCCESS" in result["formatted_output"]

    @pytest.mark.asyncio
    async def test_mcp_tool_error(self):
        """Test MCP tool with Bedrock error"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            # Mock the server components
            mock_components = MagicMock()
            mock_orchestrator = AsyncMock()
            mock_formatter = MagicMock()

            mock_orchestrator.test_bedrock_connection.return_value = {
                "status": "error",
                "region": "us-east-1",
                "error": "Access denied",
                "suggestion": "Check your AWS permissions",
            }

            mock_formatter.format_error.return_value = "❌ Error: Access denied"

            mock_components.orchestrator = mock_orchestrator
            mock_components.formatter = mock_formatter
            mock_get_components.return_value = mock_components

            result = await bedrock_connection_tool()

            assert result["status"] == "error"
            assert result["error"] == "Access denied"
            assert result["suggestion"] == "Check your AWS permissions"
            assert "formatted_output" in result
            mock_formatter.format_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tool_exception(self):
        """Test MCP tool with unexpected exception"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            # Mock the server components to raise an exception
            mock_components = MagicMock()
            mock_orchestrator = AsyncMock()
            mock_formatter = MagicMock()

            mock_orchestrator.test_bedrock_connection.side_effect = Exception("Test error")
            mock_formatter.format_error.return_value = "❌ Test failed"

            mock_components.orchestrator = mock_orchestrator
            mock_components.formatter = mock_formatter
            mock_get_components.return_value = mock_components

            result = await bedrock_connection_tool()

            assert result["status"] == "error"
            assert "Test failed: Test error" in result["error"]
            assert "Check server configuration and dependencies" in result["suggestion"]
            assert "formatted_output" in result
            mock_formatter.format_error.assert_called_once()
