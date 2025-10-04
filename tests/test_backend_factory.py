"""Tests for BackendFactory"""

from unittest.mock import MagicMock, patch

import pytest

from decision_matrix_mcp.backends import (
    BackendFactory,
    BedrockBackend,
    LiteLLMBackend,
    OllamaBackend,
)
from decision_matrix_mcp.backends.defensive import DefensiveBackendWrapper
from decision_matrix_mcp.exceptions import ConfigurationError
from decision_matrix_mcp.models import ModelBackend


class TestBackendFactory:
    """Test BackendFactory implementation"""

    @pytest.fixture()
    def factory(self):
        """Create a BackendFactory instance"""
        return BackendFactory()

    def test_factory_initialization(self, factory):
        """Test factory initialization"""
        assert len(factory._backends) == 3
        assert ModelBackend.BEDROCK in factory._backends
        assert ModelBackend.LITELLM in factory._backends
        assert ModelBackend.OLLAMA in factory._backends
        assert len(factory._instances) == 0

    def test_create_backend_bedrock(self, factory):
        """Test creating Bedrock backend wrapped in DefensiveBackendWrapper"""
        backend = factory.create_backend(ModelBackend.BEDROCK)
        assert isinstance(backend, DefensiveBackendWrapper)
        assert isinstance(backend._backend, BedrockBackend)
        assert ModelBackend.BEDROCK in factory._instances

    def test_create_backend_litellm(self, factory):
        """Test creating LiteLLM backend wrapped in DefensiveBackendWrapper"""
        backend = factory.create_backend(ModelBackend.LITELLM)
        assert isinstance(backend, DefensiveBackendWrapper)
        assert isinstance(backend._backend, LiteLLMBackend)
        assert ModelBackend.LITELLM in factory._instances

    def test_create_backend_ollama(self, factory):
        """Test creating Ollama backend wrapped in DefensiveBackendWrapper"""
        backend = factory.create_backend(ModelBackend.OLLAMA)
        assert isinstance(backend, DefensiveBackendWrapper)
        assert isinstance(backend._backend, OllamaBackend)
        assert ModelBackend.OLLAMA in factory._instances

    def test_create_backend_unknown(self, factory):
        """Test creating unknown backend type"""
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_backend("unknown_backend")
        assert "Unknown model backend" in str(exc_info.value)

    def test_backend_caching(self, factory):
        """Test that backends are cached (singleton pattern)"""
        backend1 = factory.create_backend(ModelBackend.BEDROCK)
        backend2 = factory.create_backend(ModelBackend.BEDROCK)
        assert backend1 is backend2  # Same instance

    def test_validate_backend_availability_available(self, factory):
        """Test validation when backend is available"""
        with patch.object(BedrockBackend, "is_available", return_value=True):
            result = factory.validate_backend_availability(ModelBackend.BEDROCK)
            assert result is True

    def test_validate_backend_availability_not_available(self, factory):
        """Test validation when backend is not available"""
        with patch.object(BedrockBackend, "is_available", return_value=False):
            result = factory.validate_backend_availability(ModelBackend.BEDROCK)
            assert result is False

    def test_validate_backend_availability_error(self, factory):
        """Test validation when backend creation raises error"""
        # Mock create_backend to raise an exception, and verify it's caught properly
        with patch.object(factory, "create_backend") as mock_create:
            mock_create.side_effect = Exception("Backend error")
            result = factory.validate_backend_availability(ModelBackend.BEDROCK)
            assert result is False
            mock_create.assert_called_once_with(ModelBackend.BEDROCK)

    def test_get_available_backends(self, factory):
        """Test getting list of available backends"""
        # Mock all backends as available
        with (
            patch.object(BedrockBackend, "is_available", return_value=True),
            patch.object(LiteLLMBackend, "is_available", return_value=True),
            patch.object(OllamaBackend, "is_available", return_value=True),
        ):
            available = factory.get_available_backends()
            assert len(available) == 3
            assert ModelBackend.BEDROCK in available
            assert ModelBackend.LITELLM in available
            assert ModelBackend.OLLAMA in available

    def test_get_available_backends_partial(self, factory):
        """Test getting available backends when only some are available"""
        # Mock only Bedrock as available
        with (
            patch.object(BedrockBackend, "is_available", return_value=True),
            patch.object(LiteLLMBackend, "is_available", return_value=False),
            patch.object(OllamaBackend, "is_available", return_value=False),
        ):
            available = factory.get_available_backends()
            assert len(available) == 1
            assert available == [ModelBackend.BEDROCK]

    def test_cleanup_no_instances(self, factory):
        """Test cleanup when no instances exist"""
        # Should not raise any errors
        factory.cleanup()
        assert len(factory._instances) == 0

    def test_cleanup_with_instances(self, factory):
        """Test cleanup with backend instances"""
        # Create some backends
        backend1 = factory.create_backend(ModelBackend.BEDROCK)
        backend2 = factory.create_backend(ModelBackend.LITELLM)

        assert len(factory._instances) == 2

        # Mock cleanup methods on the wrapped backends
        backend1._backend.cleanup = MagicMock()
        backend2._backend.cleanup = MagicMock()

        # Cleanup
        factory.cleanup()

        # Verify cleanup was called and instances cleared
        backend1._backend.cleanup.assert_called_once()
        backend2._backend.cleanup.assert_called_once()
        assert len(factory._instances) == 0

    def test_cleanup_with_error(self, factory):
        """Test cleanup handles errors gracefully"""
        # Create backend
        backend = factory.create_backend(ModelBackend.BEDROCK)
        backend.cleanup = MagicMock(side_effect=Exception("Cleanup error"))

        # Cleanup should not raise error
        factory.cleanup()

        # Instances should still be cleared despite error
        assert len(factory._instances) == 0

    def test_cleanup_no_cleanup_method(self, factory):
        """Test cleanup when backend doesn't have cleanup method"""
        # Create backend and remove cleanup method from the wrapped backend
        backend = factory.create_backend(ModelBackend.LITELLM)

        # Remove cleanup method from the wrapped backend, not the wrapper
        if hasattr(backend._backend, "cleanup"):
            delattr(backend._backend, "cleanup")

        # Cleanup should not raise error
        factory.cleanup()
        assert len(factory._instances) == 0
