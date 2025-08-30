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

"""Ollama backend implementation."""

import logging

from ..exceptions import LLMAPIError, LLMBackendError, LLMConfigurationError
from ..models import CriterionThread
from .base import LLMBackend

# Optional dependency imports with availability flags
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Ollama backend implementation for local LLM models."""

    @property
    def name(self) -> str:
        """Get the backend name identifier.

        Returns:
            The string "ollama" identifying this backend
        """
        return "ollama"

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming responses.

        Returns:
            False - Ollama backend does not support streaming
        """
        return False

    def is_available(self) -> bool:
        """Check if Ollama dependencies are available."""
        return HTTPX_AVAILABLE

    async def generate_response(self, thread: CriterionThread) -> str:
        """Generate response using Ollama."""
        if not HTTPX_AVAILABLE:
            raise LLMConfigurationError(
                backend="ollama",
                message="httpx is not installed. Please install with: pip install httpx",
            )

        # Import helpers here to avoid circular imports when httpx not available
        # Import config for default model
        from ..config import config
        from ..ollama_helpers import (
            build_ollama_request,
            format_messages_for_ollama,
            get_ollama_host,
            parse_ollama_response,
        )

        model = thread.criterion.model_name or config.backend.ollama_model

        try:
            # Format messages and build request
            messages = format_messages_for_ollama(thread)
            request_body = build_ollama_request(thread, messages, model)

            # Import config for timeout
            from ..config import config

            # Make API call with configurable timeout
            async with httpx.AsyncClient(timeout=config.backend.ollama_timeout_seconds) as client:
                ollama_host = get_ollama_host()
                response = await client.post(
                    f"{ollama_host}/api/chat",
                    json=request_body,
                )

                # Parse response and handle errors
                return parse_ollama_response(response, response.status_code, model)

        except ImportError as e:
            raise LLMConfigurationError(
                backend="ollama",
                message=f"httpx dependency missing: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            if isinstance(e, (LLMBackendError, LLMConfigurationError)):
                raise  # Re-raise our custom exceptions

            # Import diagnose function here to avoid circular imports
            from ..ollama_helpers import diagnose_ollama_error

            user_message = diagnose_ollama_error(e)

            raise LLMAPIError(
                backend="ollama",
                message=f"Ollama call failed: {e}",
                user_message=user_message,
                original_error=e,
            ) from e
