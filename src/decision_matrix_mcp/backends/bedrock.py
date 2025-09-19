# MIT License
#
# Copyright (c) 2025 Democratize Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AWS Bedrock backend implementation for decision criterion evaluation.

This module provides integration with AWS Bedrock service, supporting multiple
model families including Claude, Titan, and other foundation models. The backend
handles authentication, request formatting, and response parsing for the Bedrock
Converse API.

Key Features:
    - Thread-safe client management with singleton pattern
    - Comprehensive error handling with user-friendly messages
    - Support for conversation history and system prompts
    - Automatic retry and graceful degradation
    - Cost-efficient model selection and configuration
    - Structured JSON response parsing for decision scores

Configuration Requirements:
    - boto3 package installed (pip install boto3)
    - Valid AWS credentials (IAM user, role, or profile)
    - AWS_REGION or AWS_DEFAULT_REGION environment variable
    - IAM permissions: bedrock:InvokeModel, bedrock:ListFoundationModels

Supported Models:
    - Claude 3 (Sonnet, Haiku, Opus) - Recommended for analysis
    - Claude 2.1 - Legacy support
    - Amazon Titan Text - Cost-effective option
    - Other Bedrock foundation models as available

Performance Characteristics:
    - Client initialization: ~200-500ms (cached after first use)
    - Typical request latency: 1-5 seconds depending on model
    - Concurrent request limit: Based on Bedrock service quotas
    - Token costs: Varies by model (Claude 3 Sonnet ~$3-15/1M tokens)

Error Recovery:
    - Authentication failures → AWS CLI configuration guidance
    - Model access denied → Region and model availability check
    - Rate limiting → Exponential backoff and retry suggestions
    - Network timeouts → Connectivity and region recommendations

Example Usage:
    >>> backend = BedrockBackend()
    >>> if backend.is_available():
    ...     thread = CriterionThread(id, criterion, messages)
    ...     response = await backend.generate_response(thread)
    ...     # Returns structured JSON with score and justification

Troubleshooting:
    - Check AWS credentials: aws sts get-caller-identity
    - Verify region setting: echo $AWS_REGION
    - Test model access: aws bedrock list-foundation-models
    - Review IAM permissions for bedrock:InvokeModel

Note:
    All client instances use lazy initialization and are cached for
    performance. The backend includes comprehensive error diagnosis
    with actionable recovery suggestions for common configuration issues.
"""

import logging
import os
import threading
from typing import Any

from ..config import config
from ..exceptions import LLMAPIError, LLMBackendError, LLMConfigurationError
from ..models import CriterionThread
from .base import LLMBackend

# Optional dependency imports with availability flags
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Import bedrock helpers if available
if BOTO3_AVAILABLE:
    from ..bedrock_helpers import (
        build_converse_request,
        diagnose_bedrock_error,
        extract_response_text,
        format_messages_for_converse,
        get_aws_region,
    )

logger = logging.getLogger(__name__)


class BedrockBackend(LLMBackend):
    """AWS Bedrock backend for LLM-powered criterion evaluation.

    Implements the LLMBackend interface for AWS Bedrock service, providing
    access to foundation models including Claude, Titan, and others. The backend
    uses the Bedrock Converse API for unified interaction across model families.

    Features:
    - Thread-safe client caching with double-checked locking
    - Comprehensive error handling with diagnostic messages
    - Support for conversation history and structured prompts
    - Automatic model selection with sensible defaults
    - Cost optimization through efficient client reuse

    Configuration Requirements:
    - boto3 package installed (pip install boto3)
    - Valid AWS credentials configured
    - AWS_REGION or AWS_DEFAULT_REGION environment variable
    - IAM permissions for bedrock:InvokeModel

    Thread Safety:
    - Client creation protected by threading.Lock
    - Stateless design allows concurrent request processing
    - No shared mutable state between evaluations

    Example:
        >>> backend = BedrockBackend()
        >>> if backend.is_available():
        ...     response = await backend.generate_response(thread)

    Note:
        Uses singleton pattern for client instances to avoid repeated
        authentication overhead. Clients are automatically configured
        from environment variables and AWS credential chain.
    """

    def __init__(self) -> None:
        """Initialize Bedrock backend with thread-safe client management.

        Sets up client caching infrastructure but doesn't create the actual
        client until first use (lazy initialization). This avoids unnecessary
        AWS API calls during backend registration.
        """
        self._bedrock_client = None
        self._client_lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get the backend name identifier.

        Returns:
            The string "bedrock" identifying this backend
        """
        return "bedrock"

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming responses.

        Returns:
            False - Bedrock backend does not support streaming
        """
        return False

    def is_available(self) -> bool:
        """Check if AWS Bedrock backend dependencies are available.

        Verifies that the boto3 package is installed and importable.
        Does not validate AWS credentials or network connectivity.

        Returns:
            True if boto3 is available, False otherwise

        Note:
            This is a lightweight check that doesn't make AWS API calls.
            Actual connectivity and permissions are validated during
            the first generate_response() call.
        """
        return BOTO3_AVAILABLE

    def _get_bedrock_client(self) -> Any:
        """Get or create AWS Bedrock client with thread-safe singleton pattern.

        Creates a new boto3 bedrock-runtime client on first access, then
        caches it for reuse across all evaluations. Uses double-checked
        locking to ensure thread safety without performance overhead.

        Returns:
            Configured boto3 bedrock-runtime client

        Raises:
            LLMConfigurationError: If AWS credentials or region not configured

        Configuration:
            - Region: AWS_REGION or AWS_DEFAULT_REGION env vars (default: us-east-1)
            - Credentials: Uses standard AWS credential chain
            - Session: Creates new boto3.Session() for clean isolation

        Performance:
            - Client creation is expensive (~100-500ms)
            - Caching avoids repeated authentication overhead
            - Thread-safe for concurrent access patterns

        Note:
            Uses double-checked locking pattern to minimize lock contention
            while ensuring only one client instance is created.
        """
        if self._bedrock_client is None:
            with self._client_lock:
                # Double-check pattern to avoid race conditions
                if self._bedrock_client is None:
                    session = boto3.Session()
                    region = os.environ.get(
                        "AWS_REGION",
                        os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                    )
                    self._bedrock_client = session.client("bedrock-runtime", region_name=region)
        return self._bedrock_client

    async def generate_response(self, thread: CriterionThread) -> str:
        """Generate criterion evaluation response using AWS Bedrock models.

        Processes a criterion evaluation request through AWS Bedrock's Converse API,
        handling conversation history, system prompts, and model configuration.
        Provides comprehensive error handling with user-friendly diagnostics.

        Args:
            thread: CriterionThread containing conversation history and criterion settings

        Returns:
            Generated response text from the selected Bedrock model

        Raises:
            LLMConfigurationError: If boto3 not installed or AWS not configured
            LLMAPIError: If Bedrock API call fails (auth, quota, model errors)
            LLMBackendError: For unexpected errors during processing

        Model Selection:
            - Uses thread.criterion.model_name if specified
            - Defaults to Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0)
            - Supports all Bedrock-available models (Claude, Titan, etc.)

        Request Configuration:
            - System prompt from criterion configuration
            - Conversation history from thread messages
            - Temperature and max_tokens from criterion settings
            - Inference parameters optimized for decision evaluation

        Error Handling:
            - Authentication errors → Clear setup instructions
            - Model access errors → Available model suggestions
            - Quota exceeded → Retry timing recommendations
            - Network errors → Connectivity troubleshooting

        Performance:
            - Client reuse eliminates authentication overhead
            - Efficient message formatting reduces token usage
            - Concurrent requests supported (thread-safe client)

        Example:
            >>> thread = CriterionThread(id, criterion, history)
            >>> response = await backend.generate_response(thread)
            >>> # Returns: '{"score": 7, "justification": "Good performance..."}'

        Note:
            All API errors include diagnostic information and recovery
            suggestions to help users resolve configuration issues.
        """
        if not BOTO3_AVAILABLE:
            raise LLMConfigurationError(
                backend="bedrock",
                message="boto3 is not installed. Please install with: pip install boto3",
                user_message="AWS Bedrock requires boto3. Install: pip install boto3",
            )

        model_id = thread.criterion.model_name or config.backend.bedrock_model

        try:
            # Setup AWS client and prepare request
            bedrock_client = self._get_bedrock_client()

            # Format messages and build request
            messages = format_messages_for_converse(thread)
            converse_kwargs = build_converse_request(thread, messages, model_id)

            # Make API call
            response = bedrock_client.converse(**converse_kwargs)

            # Extract and return response text
            return extract_response_text(response)

        except ImportError as e:
            raise LLMConfigurationError(
                backend="bedrock",
                message=f"boto3 dependency missing: {e}",
                original_error=e,
            ) from e
        except (BotoCoreError, ClientError) as e:
            # Diagnose error and get user-friendly message
            region = get_aws_region()
            user_message, log_details = diagnose_bedrock_error(e, model_id, region)

            logger.exception("Bedrock API error: %s", log_details)

            raise LLMAPIError(
                backend="bedrock",
                message=f"Bedrock API call failed: {e}",
                user_message=user_message,
                original_error=e,
            ) from e
        except Exception as e:
            # Only catch truly unexpected errors
            if isinstance(e, (LLMBackendError, LLMConfigurationError)):
                raise  # Re-raise our custom exceptions
            raise LLMBackendError(
                backend="bedrock",
                message=f"Unexpected error in Bedrock call: {e}",
                user_message="An unexpected error occurred",
                original_error=e,
            ) from e

    def cleanup(self) -> None:
        """Clean up AWS Bedrock client resources during shutdown.

        Safely releases the cached boto3 client instance to ensure proper
        resource cleanup during application shutdown. Uses thread-safe
        operations to handle concurrent cleanup requests.

        Thread Safety:
            - Protected by the same lock used for client creation
            - Safe to call multiple times (idempotent)
            - No-op if client was never created

        Resource Management:
            - boto3 clients don't have explicit close() methods
            - Clearing reference aids garbage collection
            - Prevents resource leaks in long-running applications

        Note:
            Called automatically by BackendFactory.cleanup() during
            application shutdown. Manual calls are safe but not required.
        """
        if self._bedrock_client:
            with self._client_lock:
                if self._bedrock_client:
                    # boto3 clients don't have an explicit close method,
                    # but we can help the garbage collector by clearing the reference
                    self._bedrock_client = None
                    logger.info("Bedrock client cleaned up")
