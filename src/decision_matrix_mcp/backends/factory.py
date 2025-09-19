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

"""Factory for creating LLM backend instances."""

import logging
from typing import Any

from ..exceptions import ConfigurationError
from ..models import ModelBackend
from .base import LLMBackend
from .bedrock import BedrockBackend
from .defensive import DefensiveBackendWrapper
from .litellm import LiteLLMBackend
from .ollama import OllamaBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory for creating LLM backend instances with singleton pattern."""

    def __init__(self) -> None:
        # Registry of backend types to their implementation classes
        self._backends: dict[ModelBackend, type[LLMBackend]] = {
            ModelBackend.BEDROCK: BedrockBackend,
            ModelBackend.LITELLM: LiteLLMBackend,
            ModelBackend.OLLAMA: OllamaBackend,
        }
        # Singleton instances of backends
        self._instances: dict[ModelBackend, LLMBackend] = {}

    def create_backend(self, backend_type: ModelBackend) -> LLMBackend:
        """Create or return cached backend instance.

        Args:
            backend_type: The type of backend to create

        Returns:
            LLMBackend instance for the specified type (wrapped in defensive handler)

        Raises:
            ConfigurationError: If backend type is unknown
        """
        if backend_type not in self._backends:
            raise ConfigurationError(
                f"Unknown model backend: {backend_type}",
                f"Model backend '{backend_type}' is not registered in factory",
            )

        # Return cached instance if available
        if backend_type in self._instances:
            return self._instances[backend_type]

        # Create new instance
        backend_class = self._backends[backend_type]
        raw_instance = backend_class()

        # Always wrap in defensive handler for additional error handling
        wrapped_instance = self._ensure_defensive_wrapper(raw_instance)

        # Cache the wrapped instance
        self._instances[backend_type] = wrapped_instance

        logger.debug("Created new %s backend instance (with defensive wrapper)", backend_type.value)
        return wrapped_instance

    def _ensure_defensive_wrapper(self, backend: Any) -> LLMBackend:
        """Ensure backend is wrapped with defensive exception handling.

        Args:
            backend: The backend instance to potentially wrap

        Returns:
            Backend instance wrapped with defensive error handling
        """
        # If already wrapped, return as-is
        if isinstance(backend, DefensiveBackendWrapper):
            return backend  # type: ignore[return-value]

        # Wrap in defensive handler
        return DefensiveBackendWrapper(backend)  # type: ignore[return-value]

    def validate_backend_availability(self, backend_type: ModelBackend) -> bool:
        """Check if backend dependencies are available.

        Args:
            backend_type: The type of backend to check

        Returns:
            True if backend dependencies are available
        """
        try:
            backend = self.create_backend(backend_type)
            return backend.is_available()
        except Exception as e:  # noqa: BLE001
            logger.warning("Backend %s not available: %s", backend_type.value, e)
            return False

    def get_available_backends(self) -> list[ModelBackend]:
        """Get list of backends with available dependencies.

        Returns:
            List of backend types that have their dependencies available
        """
        return [
            backend_type
            for backend_type in self._backends
            if self.validate_backend_availability(backend_type)
        ]

    def cleanup(self) -> None:
        """Clean up all backend instances."""
        for backend_type, instance in self._instances.items():
            if hasattr(instance, "cleanup"):
                try:
                    instance.cleanup()
                    logger.debug("Cleaned up %s backend", backend_type.value)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error cleaning up %s backend: %s", backend_type.value, e)

        self._instances.clear()
