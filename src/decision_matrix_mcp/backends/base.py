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

"""Abstract base class for LLM backend implementations."""

from abc import ABC, abstractmethod

from ..models import CriterionThread


class LLMBackend(ABC):
    """Abstract base class for all LLM backend implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the backend name identifier.

        Returns:
            String identifier for this backend (e.g., "bedrock", "litellm", "ollama")
        """

    @abstractmethod
    async def generate_response(self, thread: CriterionThread) -> str:
        """Generate response for evaluation request.

        Args:
            thread: The criterion thread containing conversation history and settings

        Returns:
            The generated response text

        Raises:
            LLMConfigurationError: If backend dependencies are not available
            LLMAPIError: If the API call fails
            LLMBackendError: For other backend-specific errors
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available.

        Returns:
            True if all required dependencies are installed and configured
        """
