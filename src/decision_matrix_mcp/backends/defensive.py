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

"""Defensive programming utilities for backend error handling.

This module provides additional safety layers to ensure that no raw exceptions
escape from backend implementations, providing comprehensive defensive programming
patterns for reliable error handling.
"""

from collections.abc import Callable
import functools
import logging
from typing import Any, TypeVar

from ..exceptions import DecisionMatrixError, LLMBackendError
from ..models import CriterionThread

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def defensive_exception_wrapper(backend_name: str) -> Callable[[F], F]:
    """Decorator that provides defensive exception wrapping for backend methods.

    This decorator acts as a safety net to catch any raw exceptions that might
    escape from backend implementations due to bugs, edge cases, or unexpected
    conditions. It ensures that all exceptions are properly wrapped in
    LLMBackendError instances with appropriate context and error classification.

    Args:
        backend_name: Name of the backend for error context

    Returns:
        Decorator function that wraps exceptions defensively

    Example:
        @defensive_exception_wrapper("bedrock")
        async def generate_response(self, thread: CriterionThread) -> str:
            # Backend implementation
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except DecisionMatrixError:
                # Re-raise our custom exceptions unchanged
                raise
            except ConnectionError as e:
                logger.warning("Connection error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Connection failed: {e}",
                    user_message="Network connection failed, please check connectivity",
                    original_error=e,
                    context={"error_type": "connection", "operation": func.__name__},
                ) from e
            except TimeoutError as e:
                logger.warning("Timeout error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Request timeout: {e}",
                    user_message="Request timed out, please try again",
                    original_error=e,
                    context={"error_type": "timeout", "operation": func.__name__},
                ) from e
            except ValueError as e:
                logger.warning("Value error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Invalid value: {e}",
                    user_message="Invalid input or response format",
                    original_error=e,
                    context={"error_type": "invalid", "operation": func.__name__},
                ) from e
            except RuntimeError as e:
                logger.warning("Runtime error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Runtime error: {e}",
                    user_message="Runtime error occurred",
                    original_error=e,
                    context={"error_type": "runtime", "operation": func.__name__},
                ) from e
            except Exception as e:
                logger.exception("Unexpected error in %s backend", backend_name)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Unexpected error: {e}",
                    user_message="An unexpected error occurred",
                    original_error=e,
                    context={"error_type": "error", "operation": func.__name__},
                ) from e

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except DecisionMatrixError:
                # Re-raise our custom exceptions unchanged
                raise
            except ConnectionError as e:
                logger.warning("Connection error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Connection failed: {e}",
                    user_message="Network connection failed, please check connectivity",
                    original_error=e,
                    context={"error_type": "connection", "operation": func.__name__},
                ) from e
            except TimeoutError as e:
                logger.warning("Timeout error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Request timeout: {e}",
                    user_message="Request timed out, please try again",
                    original_error=e,
                    context={"error_type": "timeout", "operation": func.__name__},
                ) from e
            except ValueError as e:
                logger.warning("Value error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Invalid value: {e}",
                    user_message="Invalid input or response format",
                    original_error=e,
                    context={"error_type": "invalid", "operation": func.__name__},
                ) from e
            except RuntimeError as e:
                logger.warning("Runtime error in %s backend: %s", backend_name, e)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Runtime error: {e}",
                    user_message="Runtime error occurred",
                    original_error=e,
                    context={"error_type": "runtime", "operation": func.__name__},
                ) from e
            except Exception as e:
                logger.exception("Unexpected error in %s backend", backend_name)
                raise LLMBackendError(
                    backend=backend_name,
                    message=f"Unexpected error: {e}",
                    user_message="An unexpected error occurred",
                    original_error=e,
                    context={"error_type": "error", "operation": func.__name__},
                ) from e

        # Return async wrapper if the function is a coroutine
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


class DefensiveBackendWrapper:
    """Defensive wrapper for backend instances to provide additional error handling.

    This wrapper provides an additional layer of defensive programming around
    backend instances, ensuring that no raw exceptions escape even if there are
    bugs or edge cases in the backend implementations themselves.

    Usage:
        backend = DefensiveBackendWrapper(original_backend)
        response = await backend.generate_response(thread)
    """

    def __init__(self, backend: Any) -> None:
        """Initialize defensive wrapper around a backend instance.

        Args:
            backend: The original backend instance to wrap
        """
        self._backend = backend
        self._backend_name = getattr(backend, "name", "unknown")

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to wrapped backend."""
        return getattr(self._backend, name)

    def __delattr__(self, name: str) -> None:
        """Proxy attribute deletion to wrapped backend."""
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            delattr(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute setting to wrapped backend or self."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._backend, name, value)

    @property
    def __class__(self) -> type:
        """Return the wrapped backend's class for isinstance checks."""
        return self._backend.__class__

    @property
    def name(self) -> str:
        """Get the backend name."""
        return self._backend.name

    @property
    def supports_streaming(self) -> bool:
        """Check if backend supports streaming."""
        return getattr(self._backend, "supports_streaming", False)

    def is_available(self) -> bool:
        """Check if backend is available."""
        try:
            return self._backend.is_available()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error checking availability for %s backend: %s", self._backend_name, e)
            return False

    async def generate_response(self, thread: CriterionThread) -> str:
        """Generate response with defensive exception handling.

        Args:
            thread: The criterion thread to process

        Returns:
            Generated response text

        Raises:
            LLMBackendError: For any errors encountered (raw exceptions are wrapped)
        """
        try:
            return await self._backend.generate_response(thread)
        except DecisionMatrixError:
            # Re-raise our custom exceptions unchanged
            raise
        except ConnectionError as e:
            logger.warning("Connection error in %s backend: %s", self._backend_name, e)
            raise LLMBackendError(
                backend=self._backend_name,
                message=f"Connection failed: {e}",
                user_message="Network connection failed, please check connectivity",
                original_error=e,
                context={"error_type": "connection", "operation": "generate_response"},
            ) from e
        except TimeoutError as e:
            logger.warning("Timeout error in %s backend: %s", self._backend_name, e)
            raise LLMBackendError(
                backend=self._backend_name,
                message=f"Request timeout: {e}",
                user_message="Request timed out, please try again",
                original_error=e,
                context={"error_type": "timeout", "operation": "generate_response"},
            ) from e
        except ValueError as e:
            logger.warning("Value error in %s backend: %s", self._backend_name, e)
            raise LLMBackendError(
                backend=self._backend_name,
                message=f"Invalid value: {e}",
                user_message="Invalid input or response format",
                original_error=e,
                context={"error_type": "invalid", "operation": "generate_response"},
            ) from e
        except RuntimeError as e:
            logger.warning("Runtime error in %s backend: %s", self._backend_name, e)
            raise LLMBackendError(
                backend=self._backend_name,
                message=f"Runtime error: {e}",
                user_message="Runtime error occurred",
                original_error=e,
                context={"error_type": "runtime", "operation": "generate_response"},
            ) from e
        except Exception as e:
            logger.exception("Unexpected error in %s backend", self._backend_name)
            raise LLMBackendError(
                backend=self._backend_name,
                message=f"Unexpected error: {e}",
                user_message="An unexpected error occurred",
                original_error=e,
                context={"error_type": "error", "operation": "generate_response"},
            ) from e

    def cleanup(self) -> None:
        """Clean up backend resources."""
        if hasattr(self._backend, "cleanup"):
            try:
                self._backend.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.warning("Error during cleanup for %s backend: %s", self._backend_name, e)
