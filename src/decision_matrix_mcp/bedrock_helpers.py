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

"""Helper functions for AWS Bedrock integration."""

import logging
import os
from typing import Any

from .exceptions import LLMAPIError
from .models import CriterionThread

logger = logging.getLogger(__name__)


def get_aws_region() -> str:
    """Get AWS region from environment variables with fallback.

    Returns:
        AWS region string
    """
    return os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def format_messages_for_converse(thread: CriterionThread) -> list[dict[str, Any]]:
    """Format thread messages for Bedrock converse API.

    Ensures AWS Bedrock API compliance by:
    - Filtering out system messages (they belong in the system parameter)
    - Ensuring conversations start with a user message
    - Handling empty conversations gracefully

    Args:
        thread: CriterionThread containing conversation history

    Returns:
        List of formatted messages for converse API that start with a user message
    """
    # Filter out system messages - they should only be in the system parameter
    filtered_messages = [msg for msg in thread.conversation_history if msg.get("role") != "system"]

    # Format messages for Bedrock converse API
    formatted_messages = [
        {"role": msg["role"], "content": [{"text": msg["content"]}]} for msg in filtered_messages
    ]

    # AWS Bedrock requires conversations to start with a user message
    # If we don't have any messages or the first message isn't from user, add a default user message
    if not formatted_messages or formatted_messages[0]["role"] != "user":
        # Insert a default user message at the beginning
        formatted_messages.insert(
            0, {"role": "user", "content": [{"text": "Please help me evaluate this option."}]}
        )

    return formatted_messages


def build_converse_request(
    thread: CriterionThread,
    messages: list[dict[str, Any]],
    model_id: str,
) -> dict[str, Any]:
    """Build the complete converse API request.

    Args:
        thread: CriterionThread with criterion config
        messages: Formatted messages
        model_id: Bedrock model ID

    Returns:
        Complete request kwargs for converse API
    """
    return {
        "modelId": model_id,
        "messages": messages,
        "system": [{"text": thread.criterion.system_prompt}],
        "inferenceConfig": {
            "maxTokens": thread.criterion.max_tokens,
            "temperature": thread.criterion.temperature,
        },
    }


def extract_response_text(response: dict[str, Any]) -> str:
    """Extract text from Bedrock converse API response.

    Args:
        response: Response from Bedrock converse API

    Returns:
        Extracted text content

    Raises:
        LLMAPIError: If response format is invalid
    """
    if "output" in response and "message" in response["output"]:
        message_content = response["output"]["message"]["content"]
        if message_content and len(message_content) > 0:
            return message_content[0]["text"]  # type: ignore[no-any-return]

    raise LLMAPIError(
        backend="bedrock",
        message=f"Invalid response format from Bedrock converse API: {response}",
        user_message="Unexpected response format from LLM",
    )


def diagnose_bedrock_error(error: Exception, model_id: str, region: str) -> tuple[str, str]:
    """Diagnose Bedrock API errors and provide user-friendly messages.

    Args:
        error: The exception from Bedrock API
        model_id: Model ID being used
        region: AWS region

    Returns:
        Tuple of (user_message, log_details)
    """
    error_message = str(error)
    error_code = getattr(error, "response", {}).get("Error", {}).get("Code", "Unknown")

    # Log details for diagnostics
    log_details = (
        f"Bedrock API error - Code: {error_code}, "
        f"Message: {error_message}, Model: {model_id}, Region: {region}"
    )

    # Determine user-friendly message
    error_lower = error_message.lower()

    if "rate limit" in error_lower or "throttling" in error_lower:
        user_message = "Request rate limit exceeded, please try again later"
    elif "invalid" in error_lower and "model" in error_lower:
        user_message = f"Invalid model ID: {model_id}"
    elif "access" in error_lower or "permission" in error_lower:
        user_message = (
            f"No access to model {model_id} in region {region}. "
            "Check Bedrock model access in AWS Console."
        )
    elif "region" in error_lower:
        user_message = f"Bedrock not available in region {region}. Try us-east-1 or us-west-2."
    else:
        user_message = "LLM service temporarily unavailable"

    return user_message, log_details
