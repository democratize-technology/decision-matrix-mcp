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

"""LiteLLM backend implementation."""

import logging

from ..exceptions import LLMAPIError, LLMBackendError, LLMConfigurationError
from ..models import CriterionThread
from .base import LLMBackend

# Optional dependency imports with availability flags
try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class LiteLLMBackend(LLMBackend):
    """LiteLLM backend implementation for OpenAI, Anthropic, and other providers."""

    def is_available(self) -> bool:
        """Check if LiteLLM dependencies are available."""
        return LITELLM_AVAILABLE

    async def generate_response(self, thread: CriterionThread) -> str:
        """Generate response using LiteLLM."""
        if not LITELLM_AVAILABLE:
            raise LLMConfigurationError(
                backend="litellm",
                message="litellm is not installed. Please install with: pip install litellm",
            )

        try:
            # Prepare messages
            messages = [{"role": "system", "content": thread.criterion.system_prompt}]
            messages.extend(
                {"role": msg["role"], "content": msg["content"]}
                for msg in thread.conversation_history
            )

            # Choose model
            # Import config for default model
            from ..config import config

            model = thread.criterion.model_name or config.backend.litellm_model

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=thread.criterion.temperature,
                max_tokens=thread.criterion.max_tokens,
            )

            return response.choices[0].message.content

        except ImportError as e:
            raise LLMConfigurationError(
                backend="litellm",
                message=f"litellm dependency missing: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            # Check for specific error types
            error_message = str(e)
            if "rate limit" in error_message.lower() or "quota" in error_message.lower():
                user_message = "API rate limit exceeded, please try again later"
            elif "api key" in error_message.lower() or "authentication" in error_message.lower():
                user_message = "API authentication failed, check your API key"
            elif "model" in error_message.lower() and "not found" in error_message.lower():
                # Import config for default model
                from ..config import config

                model = thread.criterion.model_name or config.backend.litellm_model
                user_message = f"Model not available: {model}"
            elif isinstance(e, (LLMBackendError, LLMConfigurationError)):
                raise  # Re-raise our custom exceptions
            else:
                user_message = "LLM service temporarily unavailable"

            raise LLMAPIError(
                backend="litellm",
                message=f"LiteLLM API call failed: {e}",
                user_message=user_message,
                original_error=e,
            ) from e
