"""Tests for Bedrock connectivity testing functionality"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from decision_matrix_mcp import test_aws_bedrock_connection as bedrock_connection_tool
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestBedrockConnectivityOrchestrator:
    """Test the orchestrator's Bedrock connectivity test method"""

    @pytest.fixture()
    def orchestrator(self):
        return DecisionOrchestrator()

    @pytest.mark.asyncio()
    async def test_bedrock_connection_success(self, orchestrator):
        """Test successful Bedrock connection"""
        # Mock the backend factory methods to avoid real AWS calls
        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
            patch.dict("os.environ", {"AWS_REGION": "us-west-2"}),
        ):
            # Mock backend response
            mock_backend = AsyncMock()
            mock_backend.generate_response.return_value = "Hello!"
            mock_create_backend.return_value = mock_backend

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "ok"
            assert result["region"] == "us-west-2"
            assert result["model_tested"] == "anthropic.claude-3-haiku-20240307-v1:0"
            assert result["response_length"] == 6  # len("Hello!")
            assert result["message"] == "Bedrock connection successful"
            assert result["api_version"] == "converse"

            # Verify the backend was called correctly
            mock_create_backend.assert_called_once()
            mock_backend.generate_response.assert_called_once()

    @pytest.mark.asyncio()
    async def test_bedrock_connection_boto3_unavailable(self, orchestrator):
        """Test when boto3 is not available"""
        with patch.object(
            orchestrator.backend_factory, "validate_backend_availability", return_value=False
        ):
            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert result["error"] == "boto3 not installed. Install with: pip install boto3"
            assert result["region"] == "N/A"

    @pytest.mark.asyncio()
    async def test_bedrock_connection_access_denied(self, orchestrator):
        """Test access denied error"""
        from botocore.exceptions import ClientError

        from decision_matrix_mcp.exceptions import LLMAPIError

        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "You don't have access to the model with the specified model ID.",
            },
        }
        client_error = ClientError(error_response, "InvokeModel")
        backend_error = LLMAPIError(
            backend="bedrock",
            message="Bedrock API call failed: An error occurred (AccessDeniedException) when calling the InvokeModel operation: You don't have access to the model with the specified model ID.",
            user_message="Access denied to Bedrock model. Enable model access in AWS Console: Bedrock > Model access > Manage model access",
            original_error=client_error,
        )
        backend_error.original_error = client_error

        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
            patch.dict("os.environ", {"AWS_REGION": "us-east-1"}),
        ):
            # Mock backend to raise the API error
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = backend_error
            mock_create_backend.return_value = mock_backend

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert result["region"] == "us-east-1"
            assert result["error_code"] == "AccessDeniedException"
            assert "access" in result["error"].lower()
            assert "Enable model access in AWS Console" in result["suggestion"]

    @pytest.mark.asyncio()
    async def test_bedrock_connection_region_error(self, orchestrator):
        """Test region not available error"""
        from botocore.exceptions import ClientError

        from decision_matrix_mcp.exceptions import LLMAPIError

        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "The requested region is not supported",
            },
        }
        client_error = ClientError(error_response, "InvokeModel")
        backend_error = LLMAPIError(
            backend="bedrock",
            message="Bedrock API call failed: An error occurred (ValidationException) when calling the InvokeModel operation: The requested region is not supported",
            user_message="Region not supported for Bedrock. Try us-east-1 or us-west-2 regions where Bedrock is available",
            original_error=client_error,
        )
        backend_error.original_error = client_error

        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
        ):
            # Mock backend to raise the API error
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = backend_error
            mock_create_backend.return_value = mock_backend

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "region" in result["error"].lower()
            assert "Try us-east-1 or us-west-2" in result["suggestion"]

    @pytest.mark.asyncio()
    async def test_bedrock_connection_credentials_error(self, orchestrator):
        """Test credentials error"""
        from botocore.exceptions import ClientError

        from decision_matrix_mcp.exceptions import LLMAPIError

        error_response = {
            "Error": {"Code": "UnauthorizedOperation", "Message": "Unable to locate credentials"},
        }
        client_error = ClientError(error_response, "InvokeModel")
        backend_error = LLMAPIError(
            backend="bedrock",
            message="Bedrock API call failed: An error occurred (UnauthorizedOperation) when calling the InvokeModel operation: Unable to locate credentials",
            user_message="AWS credentials not configured. Configure AWS credentials: aws configure or set AWS_PROFILE environment variable",
            original_error=client_error,
        )
        backend_error.original_error = client_error

        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
        ):
            # Mock backend to raise the API error
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = backend_error
            mock_create_backend.return_value = mock_backend

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "credentials" in result["error"].lower()
            assert "Configure AWS credentials" in result["suggestion"]

    @pytest.mark.asyncio()
    async def test_bedrock_connection_throttling_error(self, orchestrator):
        """Test throttling error"""
        from botocore.exceptions import ClientError

        from decision_matrix_mcp.exceptions import LLMAPIError

        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        client_error = ClientError(error_response, "InvokeModel")
        backend_error = LLMAPIError(
            backend="bedrock",
            message="Bedrock API call failed: An error occurred (ThrottlingException) when calling the InvokeModel operation: Rate exceeded",
            user_message="Bedrock API rate limit exceeded. Request rate limit exceeded. Wait a moment and try again",
            original_error=client_error,
        )
        backend_error.original_error = client_error

        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
        ):
            # Mock backend to raise the API error
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = backend_error
            mock_create_backend.return_value = mock_backend

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "throttling" in result["error"].lower()
            assert "Request rate limit exceeded" in result["suggestion"]

    @pytest.mark.asyncio()
    async def test_bedrock_connection_unexpected_error(self, orchestrator):
        """Test unexpected error"""
        from decision_matrix_mcp.exceptions import LLMBackendError

        unexpected_error = Exception("Unexpected error")
        backend_error = LLMBackendError(
            backend="bedrock",
            message="Unexpected error in Bedrock call: Unexpected error",
            user_message="An unexpected error occurred",
            original_error=unexpected_error,
        )

        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
        ):
            # Mock backend to raise unexpected error
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = backend_error
            mock_create_backend.return_value = mock_backend

            result = await orchestrator.test_bedrock_connection()

            assert result["status"] == "error"
            assert "Unexpected error" in result["error"]
            assert "Check AWS Bedrock service status" in result["suggestion"]

    @pytest.mark.asyncio()
    async def test_bedrock_connection_default_region(self, orchestrator):
        """Test default region is used when no environment variable set"""
        with (
            patch.object(
                orchestrator.backend_factory, "validate_backend_availability", return_value=True
            ),
            patch.object(orchestrator.backend_factory, "create_backend") as mock_create_backend,
            patch.dict("os.environ", {}, clear=True),
        ):
            # Mock backend response
            mock_backend = AsyncMock()
            mock_backend.generate_response.return_value = "Hello!"
            mock_create_backend.return_value = mock_backend

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
            "ValidationException",
            "region not available",
        )
        assert "Try us-east-1 or us-west-2" in suggestion

        # Credentials error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "UnauthorizedOperation",
            "credentials not found",
        )
        assert "Configure AWS credentials" in suggestion

        # Throttling error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "ThrottlingException",
            "throttling limit exceeded",
        )
        assert "Request rate limit exceeded" in suggestion

        # Unknown error
        suggestion = orchestrator._get_bedrock_error_suggestion(
            "UnknownError",
            "something went wrong",
        )
        assert "Check AWS Bedrock service status" in suggestion


class TestBedrockConnectivityMCPTool:
    """Test the MCP tool wrapper for Bedrock connectivity testing"""

    @pytest.fixture()
    def mock_ctx(self):
        """Mock context for MCP tool calls"""
        return Mock()

    @pytest.mark.asyncio()
    async def test_mcp_tool_success(self, mock_ctx):
        """Test successful MCP tool call"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            # Mock the server components
            mock_components = MagicMock()
            mock_decision_service = AsyncMock()
            mock_response_service = MagicMock()

            test_result = {
                "status": "ok",
                "region": "us-west-2",
                "model_tested": "anthropic.claude-3-haiku-20240307-v1:0",
                "response_length": 6,
                "message": "Bedrock connection successful",
            }

            mock_decision_service.test_bedrock_connection.return_value = test_result
            mock_response_service.create_bedrock_test_response.return_value = test_result

            mock_components.decision_service = mock_decision_service
            mock_components.response_service = mock_response_service
            mock_get_components.return_value = mock_components

            result = await bedrock_connection_tool(ctx=mock_ctx)

            assert result["status"] == "ok"
            assert result["region"] == "us-west-2"
            assert result["model_tested"] == "anthropic.claude-3-haiku-20240307-v1:0"
            assert result["message"] == "Bedrock connection successful"

    @pytest.mark.asyncio()
    async def test_mcp_tool_error(self, mock_ctx):
        """Test MCP tool with Bedrock error"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            # Mock the server components
            mock_components = MagicMock()
            mock_decision_service = AsyncMock()
            mock_response_service = MagicMock()

            test_result = {
                "status": "error",
                "region": "us-east-1",
                "error": "Access denied",
                "suggestion": "Check your AWS permissions",
            }

            mock_decision_service.test_bedrock_connection.return_value = test_result
            mock_response_service.create_bedrock_test_response.return_value = test_result

            mock_components.decision_service = mock_decision_service
            mock_components.response_service = mock_response_service
            mock_get_components.return_value = mock_components

            result = await bedrock_connection_tool(ctx=mock_ctx)

            assert result["status"] == "error"
            assert result["error"] == "Access denied"
            assert result["suggestion"] == "Check your AWS permissions"

    @pytest.mark.asyncio()
    async def test_mcp_tool_exception(self, mock_ctx):
        """Test MCP tool with unexpected exception"""
        with patch("decision_matrix_mcp.get_server_components") as mock_get_components:
            # Mock the server components to raise an exception
            mock_components = MagicMock()
            mock_decision_service = AsyncMock()
            mock_response_service = MagicMock()

            # Make the decision service raise an exception
            mock_decision_service.test_bedrock_connection.side_effect = Exception("Test error")

            # Mock the error response
            error_response = {
                "status": "error",
                "error": "Test failed: Test error",
                "user_message": "Bedrock test error",
            }
            mock_response_service.create_error_response.return_value = error_response

            mock_components.decision_service = mock_decision_service
            mock_components.response_service = mock_response_service
            mock_get_components.return_value = mock_components

            result = await bedrock_connection_tool(ctx=mock_ctx)

            assert result["status"] == "error"
            assert result["error"] == "Test failed: Test error"
            assert result["user_message"] == "Bedrock test error"
