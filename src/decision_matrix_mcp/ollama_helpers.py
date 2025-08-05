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

"""Helper functions for Ollama integration"""

import os
from typing import Any

from .exceptions import LLMAPIError
from .models import CriterionThread


def get_ollama_host() -> str:
    """Get Ollama host from environment with default
    
    Returns:
        Ollama host URL
    """
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def format_messages_for_ollama(thread: CriterionThread) -> list[dict[str, str]]:
    """Format thread messages for Ollama API
    
    Args:
        thread: CriterionThread containing conversation history
        
    Returns:
        List of formatted messages for Ollama API
    """
    messages = [{"role": "system", "content": thread.criterion.system_prompt}]
    for msg in thread.conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def build_ollama_request(
    thread: CriterionThread,
    messages: list[dict[str, str]],
    model: str
) -> dict[str, Any]:
    """Build the complete Ollama API request
    
    Args:
        thread: CriterionThread with criterion config
        messages: Formatted messages
        model: Ollama model name
        
    Returns:
        Complete request body for Ollama API
    """
    options = {
        "temperature": thread.criterion.temperature,
        "num_ctx": 4096
    }
    
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }


def parse_ollama_response(
    response: Any,
    status_code: int,
    model: str
) -> str:
    """Parse Ollama API response and handle errors
    
    Args:
        response: HTTP response object
        status_code: HTTP status code
        model: Model name for error messages
        
    Returns:
        Extracted response text
        
    Raises:
        LLMAPIError: If response indicates an error
    """
    if status_code != 200:
        error_msg = f"Ollama API error: {status_code}"
        
        # Try to extract error details from response
        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg = f"Ollama API error: {error_data['error']}"
        except Exception:
            pass
        
        # Provide specific error messages based on status code
        if status_code == 404:
            raise LLMAPIError(
                backend="ollama",
                message=error_msg,
                user_message=f"Model not available in Ollama: {model}",
            )
        elif status_code == 503:
            raise LLMAPIError(
                backend="ollama",
                message=error_msg,
                user_message="Ollama service is not running",
            )
        else:
            raise LLMAPIError(
                backend="ollama",
                message=error_msg,
                user_message="Ollama service temporarily unavailable",
            )
    
    result = response.json()
    return result["message"]["content"]


def diagnose_ollama_error(error: Exception) -> str:
    """Diagnose Ollama connection errors and provide user-friendly messages
    
    Args:
        error: The exception from Ollama API
        
    Returns:
        User-friendly error message
    """
    error_message = str(error).lower()
    
    if "connection" in error_message or "refused" in error_message:
        return "Cannot connect to Ollama service. Is it running?"
    else:
        return "Ollama service error"