"""Integration tests for HTTP transport.

Tests the MCP Spec 2025-06-18 Streamable HTTP transport implementation.
"""

import pytest
from starlette.testclient import TestClient


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

    @pytest.fixture()
    def client(self, http_app):
        """Create test client fixture."""
        return TestClient(http_app)

    def test_health_endpoint_exists(self, http_app):
        """Test health check endpoint is registered."""
        routes = [route.path for route in http_app.routes]
        assert "/health" in routes

    def test_mcp_endpoint_exists(self, http_app):
        """Test MCP endpoint is registered."""
        routes = [route.path for route in http_app.routes]
        assert "/mcp" in routes

    def test_health_endpoint_responds(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["transport"] == "streamable-http"


class TestHTTPJSONRPCRouting:
    """Test JSON-RPC request routing to FastMCP methods."""

    @pytest.fixture()
    def client(self):
        """Create test client fixture."""
        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        return TestClient(app)

    def test_tools_list_request(self, client):
        """Test tools/list JSON-RPC request."""
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert "tools" in data["result"]
        assert isinstance(data["result"]["tools"], list)

    def test_invalid_json_rpc_missing_method(self, client):
        """Test request with missing method field."""
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "params": {}},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert "error" in data
        assert data["error"]["code"] == -32600
        assert "missing method" in data["error"]["message"].lower()

    def test_invalid_json_rpc_non_dict_body(self, client):
        """Test request with non-dict body."""
        response = client.post(
            "/mcp",
            json="invalid",
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600

    def test_unknown_method(self, client):
        """Test request with unknown method."""
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 3, "method": "unknown/method", "params": {}},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 3
        assert "error" in data
        assert data["error"]["code"] == -32601
        assert "not found" in data["error"]["message"].lower()

    def test_tools_call_missing_name(self, client):
        """Test tools/call request without tool name."""
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"arguments": {}}},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 4
        assert "error" in data
        assert data["error"]["code"] == -32602
        assert "missing tool name" in data["error"]["message"].lower()


class TestHTTPCORS:
    """Test CORS handling in HTTP transport."""

    @pytest.fixture()
    def client(self):
        """Create test client fixture."""
        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        return TestClient(app)

    def test_cors_preflight_allowed_origin(self, client):
        """Test CORS preflight for allowed origin."""
        response = client.options(
            "/mcp",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_headers_in_response(self, client):
        """Test CORS headers are included in MCP responses."""
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
