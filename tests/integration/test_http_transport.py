"""Integration tests for HTTP transport.

Tests the MCP Spec 2025-06-18 Streamable HTTP transport implementation.
"""

import os

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


class TestCORSConfiguration:
    """Test CORS environment variable configuration."""

    def test_default_cors_origins(self):
        """Test default CORS origins when environment variable not set."""
        # Ensure MCP_CORS_ORIGINS is not set
        env_backup = os.environ.get("MCP_CORS_ORIGINS")
        if "MCP_CORS_ORIGINS" in os.environ:
            del os.environ["MCP_CORS_ORIGINS"]

        try:
            from decision_matrix_mcp.transports import create_http_app

            app = create_http_app()
            client = TestClient(app)

            # Test default localhost origins work
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
            )

            assert response.status_code == 200
            assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

        finally:
            # Restore environment
            if env_backup:
                os.environ["MCP_CORS_ORIGINS"] = env_backup

    def test_custom_cors_origins_from_env(self, monkeypatch):
        """Test custom CORS origins from environment variable."""
        # Set custom origins
        monkeypatch.setenv("MCP_CORS_ORIGINS", "https://example.com,https://app.example.com")

        # Import after setting environment variable
        import sys

        # Remove cached module to force reimport with new env
        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        client = TestClient(app)

        # Test custom origin works
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Origin": "https://example.com"},
        )

        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"

    def test_cors_rejects_unauthorized_origin(self):
        """Test CORS rejects unauthorized origins."""
        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        client = TestClient(app)

        # Test unauthorized origin is rejected
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Origin": "https://evil.com"},
        )

        # Should return 403 for invalid origin
        assert response.status_code == 403


class TestCORSValidation:
    """Test CORS origin validation security."""

    def test_wildcard_origin_rejected(self, monkeypatch):
        """Test wildcard CORS origins are rejected for security."""
        import sys

        monkeypatch.setenv("MCP_CORS_ORIGINS", "*")

        # Remove cached module to force reimport with new env
        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        # Should raise ValueError when trying to create app
        import pytest

        from decision_matrix_mcp.transports import create_http_app

        with pytest.raises(ValueError, match="Wildcard CORS origins"):
            create_http_app()

    def test_malformed_url_rejected(self, monkeypatch):
        """Test malformed URLs are rejected."""
        import sys

        monkeypatch.setenv("MCP_CORS_ORIGINS", "not-a-url,ftp://wrong-protocol.com")

        # Remove cached module to force reimport with new env
        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        # Should raise ValueError for invalid format
        import pytest

        from decision_matrix_mcp.transports import create_http_app

        with pytest.raises(ValueError, match="Invalid CORS origin format"):
            create_http_app()

    def test_empty_origins_rejected(self, monkeypatch):
        """Test empty CORS origins configuration is rejected."""
        import sys

        monkeypatch.setenv("MCP_CORS_ORIGINS", "   ,  ,  ")

        # Remove cached module to force reimport with new env
        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        # Should raise ValueError for no valid origins
        import pytest

        from decision_matrix_mcp.transports import create_http_app

        with pytest.raises(ValueError, match="No valid CORS origins"):
            create_http_app()

    def test_too_many_origins_rejected(self, monkeypatch):
        """Test excessive number of CORS origins is rejected."""
        import sys

        # Create 21 origins (exceeds MAX_CORS_ORIGINS = 20)
        many_origins = ",".join([f"https://app{i}.example.com" for i in range(21)])
        monkeypatch.setenv("MCP_CORS_ORIGINS", many_origins)

        # Remove cached module to force reimport with new env
        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        # Should raise ValueError for too many origins
        import pytest

        from decision_matrix_mcp.transports import create_http_app

        with pytest.raises(ValueError, match="Too many CORS origins"):
            create_http_app()


class TestRequestSizeLimits:
    """Test request size limit security controls."""

    @pytest.fixture()
    def client(self):
        """Create test client fixture."""
        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        return TestClient(app)

    def test_oversized_request_rejected(self, client, monkeypatch):
        """Test requests exceeding size limit are rejected."""
        # Set small limit for testing (1KB)
        monkeypatch.setenv("MCP_MAX_REQUEST_SIZE", "1024")

        # Force module reload with new env var
        import sys

        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        client = TestClient(app)

        # Create request body exceeding 1KB
        large_payload = {"data": "x" * 2000}  # ~2KB JSON

        response = client.post(
            "/mcp",
            json=large_payload,
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert "too large" in data["error"]["message"].lower()

    def test_deeply_nested_json_rejected(self, client):
        """Test JSON with excessive nesting is rejected."""
        # Create deeply nested JSON (depth > 32)
        nested = {"level": 1}
        current = nested
        for i in range(2, 40):  # Create 39 levels of nesting
            current["nested"] = {"level": i}
            current = current["nested"]

        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": nested},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "nesting too deep" in data["error"]["message"].lower()

    def test_wide_json_object_rejected(self, monkeypatch):
        """Test JSON object with too many keys is rejected."""
        # Ensure large request size limit for this test
        monkeypatch.setenv("MCP_MAX_REQUEST_SIZE", str(100 * 1024 * 1024))  # 100MB

        # Force module reload
        import sys

        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        client = TestClient(app)

        # Create object with > 1000 keys
        wide_object = {f"key_{i}": f"value_{i}" for i in range(1500)}

        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": wide_object},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "too wide" in data["error"]["message"].lower()

    def test_large_json_array_rejected(self, monkeypatch):
        """Test JSON array with too many items is rejected."""
        # Ensure large request size limit for this test
        monkeypatch.setenv("MCP_MAX_REQUEST_SIZE", str(100 * 1024 * 1024))  # 100MB

        # Force module reload
        import sys

        if "decision_matrix_mcp.transports.http_server" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports.http_server"]
        if "decision_matrix_mcp.transports" in sys.modules:
            del sys.modules["decision_matrix_mcp.transports"]

        from decision_matrix_mcp.transports import create_http_app

        app = create_http_app()
        client = TestClient(app)

        # Create array with > 1000 items
        large_array = [f"item_{i}" for i in range(1500)]

        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": large_array},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "too large" in data["error"]["message"].lower()

    def test_normal_request_accepted(self, client):
        """Test normal-sized requests are not blocked."""
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
        )

        # Should succeed (not be rejected by size limits)
        assert response.status_code == 200
