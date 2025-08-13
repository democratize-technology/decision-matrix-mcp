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

"""Tests for the service layer components"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from decision_matrix_mcp.dependency_injection import ServiceContainer
from decision_matrix_mcp.services import DecisionService, ValidationService, ResponseService
from decision_matrix_mcp.models import ModelBackend, DecisionSession
from decision_matrix_mcp.formatting import DecisionFormatter


class TestServiceContainer:
    """Test the dependency injection container"""

    def test_container_initialization(self):
        """Test that container initializes correctly"""
        container = ServiceContainer()
        assert not container._initialized

        container.initialize()
        assert container._initialized

    def test_service_creation(self):
        """Test that all services are created properly"""
        container = ServiceContainer()
        container.initialize()

        # Test core component services
        assert container.get_orchestrator() is not None
        assert container.get_session_manager() is not None
        assert container.get_formatter() is not None

        # Test service layer
        assert container.get_decision_service() is not None
        assert container.get_validation_service() is not None
        assert container.get_response_service() is not None

    def test_singleton_behavior(self):
        """Test that services are singletons within container"""
        container = ServiceContainer()
        container.initialize()

        # Same instance should be returned on multiple calls
        service1 = container.get_decision_service()
        service2 = container.get_decision_service()
        assert service1 is service2

    def test_context_manager(self):
        """Test container as context manager"""
        with ServiceContainer() as container:
            assert container._initialized
            service = container.get_decision_service()
            assert service is not None

        # Container should be cleaned up after context
        assert not container._initialized

    def test_cleanup(self):
        """Test container cleanup"""
        container = ServiceContainer()
        container.initialize()

        assert container._initialized
        container.cleanup()
        assert not container._initialized


class TestDecisionService:
    """Test the DecisionService"""

    @pytest.fixture
    def mock_session_manager(self):
        return Mock()

    @pytest.fixture
    def mock_orchestrator(self):
        return Mock()

    @pytest.fixture
    def decision_service(self, mock_session_manager, mock_orchestrator):
        return DecisionService(mock_session_manager, mock_orchestrator)

    def test_create_session(self, decision_service, mock_session_manager):
        """Test session creation delegation"""
        mock_session = Mock()
        mock_session_manager.create_session.return_value = mock_session

        result = decision_service.create_session("test topic", ["option1", "option2"])

        mock_session_manager.create_session.assert_called_once_with(
            topic="test topic", initial_options=["option1", "option2"], temperature=0.1
        )
        assert result is mock_session

    def test_get_session(self, decision_service, mock_session_manager):
        """Test session retrieval"""
        mock_session = Mock()
        mock_session_manager.get_session.return_value = mock_session

        result = decision_service.get_session("session-id")

        mock_session_manager.get_session.assert_called_once_with("session-id")
        assert result is mock_session

    def test_create_criterion_from_request(self, decision_service):
        """Test criterion creation from request"""
        request = Mock()
        request.name = "test_criterion"
        request.description = "Test description"
        request.weight = 2.0
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = "test-model"
        request.temperature = 0.5
        request.custom_prompt = "Custom prompt"

        session = Mock()
        session.default_temperature = 0.1

        criterion = decision_service.create_criterion_from_request(request, session)

        assert criterion.name == "test_criterion"
        assert criterion.description == "Test description"
        assert criterion.weight == 2.0
        assert criterion.temperature == 0.5
        assert criterion.system_prompt == "Custom prompt"

    def test_create_criterion_uses_session_default_temperature(self, decision_service):
        """Test criterion uses session default when temperature is None"""
        request = Mock()
        request.name = "test_criterion"
        request.description = "Test description"
        request.weight = 1.0
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = None
        request.temperature = None
        request.custom_prompt = None

        session = Mock()
        session.default_temperature = 0.3

        criterion = decision_service.create_criterion_from_request(request, session)
        assert criterion.temperature == 0.3

    @pytest.mark.asyncio
    async def test_execute_parallel_evaluation(self, decision_service, mock_orchestrator):
        """Test parallel evaluation delegation"""
        session = Mock()
        session.threads = {"criterion1": "thread1"}
        session.options = {"opt1": "option1"}
        session.criteria = {"criterion1": "crit1"}  # Add criteria for len() check

        mock_results = {"criterion1": {"opt1": (8.5, "Good option")}}
        mock_orchestrator.evaluate_options_across_criteria = AsyncMock(return_value=mock_results)

        result = await decision_service.execute_parallel_evaluation(session)

        mock_orchestrator.evaluate_options_across_criteria.assert_called_once_with(
            {"criterion1": "thread1"}, ["option1"]
        )
        assert result == mock_results

    @pytest.mark.asyncio
    async def test_test_bedrock_connection(self, decision_service, mock_orchestrator):
        """Test Bedrock connection test delegation"""
        mock_result = {"status": "ok", "region": "us-east-1"}
        mock_orchestrator.test_bedrock_connection = AsyncMock(return_value=mock_result)

        result = await decision_service.test_bedrock_connection()

        mock_orchestrator.test_bedrock_connection.assert_called_once()
        assert result == mock_result


class TestValidationService:
    """Test the ValidationService"""

    @pytest.fixture
    def validation_service(self):
        return ValidationService()

    def test_validate_session_id(self, validation_service):
        """Test session ID validation delegation"""
        # The ValidationService uses self.session_validator which is set to SessionValidator class
        with patch.object(
            validation_service.session_validator, "validate_session_id", return_value=True
        ) as mock_validate:
            result = validation_service.validate_session_id("test-id")

            mock_validate.assert_called_once_with("test-id")
            assert result is True

    def test_validate_session_exists_success(self, validation_service):
        """Test successful session existence validation"""
        with patch.object(
            validation_service.session_validator, "validate_session_id", return_value=True
        ):
            mock_session_manager = Mock()
            mock_session = Mock()
            mock_session_manager.get_session.return_value = mock_session

            session, error = validation_service.validate_session_exists(
                "valid-id", mock_session_manager
            )

            assert session is mock_session
            assert error is None

    def test_validate_session_exists_invalid_id(self, validation_service):
        """Test session existence validation with invalid ID"""
        with patch.object(
            validation_service.session_validator, "validate_session_id", return_value=False
        ):
            mock_session_manager = Mock()

            session, error = validation_service.validate_session_exists(
                "invalid-id", mock_session_manager
            )

            assert session is None
            assert error == {"error": "Invalid session ID format"}

    def test_validate_session_exists_not_found(self, validation_service):
        """Test session existence validation when session not found"""
        with patch.object(
            validation_service.session_validator, "validate_session_id", return_value=True
        ):
            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = None

            session, error = validation_service.validate_session_exists(
                "missing-id", mock_session_manager
            )

            assert session is None
            assert error == {"error": "Session missing-id not found or expired"}

    def test_validate_evaluation_prerequisites_no_options(self, validation_service):
        """Test evaluation prerequisites with no options"""
        session = Mock()
        session.options = {}
        session.criteria = {"criterion1": "test"}

        result = validation_service.validate_evaluation_prerequisites(session)

        assert result["error"] == "No options to evaluate. Add options first."
        assert result["validation_context"] == "prerequisites"

    def test_validate_evaluation_prerequisites_no_criteria(self, validation_service):
        """Test evaluation prerequisites with no criteria"""
        session = Mock()
        session.options = {"option1": "test"}
        session.criteria = {}

        result = validation_service.validate_evaluation_prerequisites(session)

        assert result["error"] == "No criteria defined. Add criteria first."

    def test_validate_evaluation_prerequisites_success(self, validation_service):
        """Test successful evaluation prerequisites validation"""
        session = Mock()
        session.options = {"option1": "test"}
        session.criteria = {"criterion1": "test"}

        result = validation_service.validate_evaluation_prerequisites(session)

        assert result is None


class TestResponseService:
    """Test the ResponseService"""

    @pytest.fixture
    def mock_formatter(self):
        formatter = Mock()
        formatter.format_error.return_value = "Formatted error"
        formatter.format_session_created.return_value = "Formatted session"
        return formatter

    @pytest.fixture
    def response_service(self, mock_formatter):
        return ResponseService(mock_formatter)

    def test_create_error_response(self, response_service, mock_formatter):
        """Test error response creation"""
        result = response_service.create_error_response("Test error", "Test context")

        assert result["error"] == "Test error"
        assert result["formatted_output"] == "Formatted error"
        mock_formatter.format_error.assert_called_once_with("Test error", "Test context")

    def test_create_session_response(self, response_service, mock_formatter):
        """Test session response creation"""
        session = Mock()
        session.session_id = "session-123"

        request = Mock()
        request.topic = "Test topic"
        request.options = ["option1", "option2"]
        request.model_backend.value = "bedrock"
        request.model_name = "test-model"

        criteria_added = ["criterion1"]

        result = response_service.create_session_response(session, request, criteria_added)

        assert result["session_id"] == "session-123"
        assert result["topic"] == "Test topic"
        assert result["options"] == ["option1", "option2"]
        assert result["criteria_added"] == ["criterion1"]
        assert result["formatted_output"] == "Formatted session"
        assert "Decision analysis initialized with 2 options and 1 criteria" in result["message"]

    def test_create_criterion_response(self, response_service, mock_formatter):
        """Test criterion response creation"""
        request = Mock()
        request.session_id = "session-123"
        request.name = "test_criterion"
        request.description = "Test description"
        request.weight = 2.0

        session = Mock()
        session.criteria = {"test_criterion": Mock(), "other": Mock()}

        mock_formatter.format_criterion_added.return_value = "Formatted criterion"

        result = response_service.create_criterion_response(request, session)

        assert result["session_id"] == "session-123"
        assert result["criterion_added"] == "test_criterion"
        assert result["weight"] == 2.0
        assert result["total_criteria"] == 2
        assert result["formatted_output"] == "Formatted criterion"
