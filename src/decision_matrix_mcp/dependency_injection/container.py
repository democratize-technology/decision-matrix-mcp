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

"""Service Container - Dependency injection container for all services
Provides centralized service creation and management.
"""

import logging
from typing import Any

from ..error_middleware import MCPErrorHandler
from ..formatting import DecisionFormatter
from ..orchestrator import DecisionOrchestrator
from ..services import DecisionService, ResponseService, ValidationService
from ..session_manager import SessionManager
from ..validation_decorators import ValidationErrorFormatter

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Dependency injection container for all services
    Provides singleton service instances with proper dependency wiring.
    """

    def __init__(self) -> None:
        """Initialize the service container."""
        self._services: dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all services with proper dependency injection."""
        if self._initialized:
            return

        logger.debug("Initializing service container...")

        # Core components (no dependencies)
        self._services["orchestrator"] = DecisionOrchestrator()
        self._services["session_manager"] = SessionManager()
        self._services["formatter"] = DecisionFormatter()

        # Initialize validation error formatter with the formatter
        ValidationErrorFormatter.initialize(self._services["formatter"])

        # Service layer (with dependencies)
        self._services["validation_service"] = ValidationService()
        self._services["response_service"] = ResponseService(self._services["formatter"])
        self._services["decision_service"] = DecisionService(
            self._services["session_manager"],
            self._services["orchestrator"],
        )
        self._services["error_handler"] = MCPErrorHandler(self._services["response_service"])

        self._initialized = True
        logger.debug("Service container initialized successfully")

    def get_orchestrator(self) -> DecisionOrchestrator:
        """Get the DecisionOrchestrator instance."""
        self._ensure_initialized()
        return self._services["orchestrator"]

    def get_session_manager(self) -> SessionManager:
        """Get the SessionManager instance."""
        self._ensure_initialized()
        return self._services["session_manager"]

    def get_formatter(self) -> DecisionFormatter:
        """Get the DecisionFormatter instance."""
        self._ensure_initialized()
        return self._services["formatter"]

    def get_decision_service(self) -> DecisionService:
        """Get the DecisionService instance."""
        self._ensure_initialized()
        return self._services["decision_service"]

    def get_validation_service(self) -> ValidationService:
        """Get the ValidationService instance."""
        self._ensure_initialized()
        return self._services["validation_service"]

    def get_response_service(self) -> ResponseService:
        """Get the ResponseService instance."""
        self._ensure_initialized()
        return self._services["response_service"]

    def get_error_handler(self) -> MCPErrorHandler:
        """Get the MCPErrorHandler instance."""
        self._ensure_initialized()
        return self._services["error_handler"]

    def cleanup(self) -> None:
        """Clean up all services and resources."""
        if not self._initialized:
            return

        logger.debug("Cleaning up service container...")

        # Clean up session manager
        if "session_manager" in self._services:
            self._services["session_manager"].clear_all_sessions()

        # Clean up orchestrator
        if "orchestrator" in self._services:
            self._services["orchestrator"].cleanup()

        # Clear all services
        self._services.clear()
        self._initialized = False

        logger.debug("Service container cleaned up")

    def _ensure_initialized(self) -> None:
        """Ensure the container is initialized."""
        if not self._initialized:
            self.initialize()

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
