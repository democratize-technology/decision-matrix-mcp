"""Integration tests for HTTP transport.

Tests the MCP Spec 2025-06-18 Streamable HTTP transport implementation.
"""

import pytest


class TestHTTPTransportImport:
    """Test HTTP transport module import and initialization."""

    def test_http_transport_imports_without_error(self):
        """Test that HTTP transport module can be imported."""
        # This test verifies fix for: "HTTP server attempts to import from non-existent 'core' module"
        # Previously: from ..core import mcp (ModuleNotFoundError)
        # Fixed: from .. import mcp
        from decision_matrix_mcp.transports import create_http_app

        assert create_http_app is not None
        assert callable(create_http_app)

    def test_http_app_creation(self):
        """Test that HTTP app can be created."""
        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        assert app is not None
        # Verify it's a Starlette application
        assert hasattr(app, "routes")
        assert len(app.routes) == 2  # /mcp and /health endpoints


class TestHTTPTransportEndpoints:
    """Test HTTP transport endpoint routing."""

    @pytest.fixture()
    def http_app(self):
        """Create HTTP app fixture."""
        from decision_matrix_mcp.transports import create_http_app

        return create_http_app()

    def test_health_endpoint_exists(self, http_app):
        """Test health check endpoint is registered."""
        routes = [route.path for route in http_app.routes]
        assert "/health" in routes

    def test_mcp_endpoint_exists(self, http_app):
        """Test MCP endpoint is registered."""
        routes = [route.path for route in http_app.routes]
        assert "/mcp" in routes
