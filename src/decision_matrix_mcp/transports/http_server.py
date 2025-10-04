"""HTTP server for Decision Matrix MCP.

Implements MCP Spec 2025-06-18 Streamable HTTP transport.
Self-contained implementation with no external dependencies.
"""

import json
import logging
import secrets
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def create_http_app() -> Starlette:
    """Create Starlette HTTP app for MCP server.

    Returns:
        Starlette application instance
    """
    from .. import mcp

    # Security validator
    validator = SecurityValidator()

    async def handle_mcp_request(request: Request) -> Response:  # noqa: PLR0911
        """Handle MCP requests via HTTP."""
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return _cors_preflight(request, validator)

        # Validate origin
        origin = request.headers.get("origin", "")
        if origin and not validator.validate_origin(origin):
            logger.warning("Invalid origin rejected: %s", origin)
            return JSONResponse(
                {"error": "Invalid origin"},
                status_code=403,
                headers=validator.get_cors_headers(origin),
            )

        # Validate content type
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": "Content-Type must be application/json"},
                },
                status_code=400,
                headers=validator.get_cors_headers(origin),
            )

        # Parse request body
        try:
            body = await request.json()
        except Exception:
            logger.exception("JSON parse error")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
                status_code=400,
                headers=validator.get_cors_headers(origin),
            )

        # Handle request
        try:
            # Check if streaming requested
            accept = request.headers.get("accept", "")
            if "text/event-stream" in accept and _should_stream(body):
                return await _handle_streaming(body, mcp, validator, origin)
            return await _handle_json(body, mcp, validator, origin)
        except Exception:
            logger.exception("Request handling error")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": body.get("id") if isinstance(body, dict) else None,
                    "error": {"code": -32603, "message": "Internal error"},
                },
                status_code=500,
                headers=validator.get_cors_headers(origin),
            )

    async def health_check(_request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            {"status": "ok", "transport": "streamable-http", "server": "decision-matrix-mcp"}
        )

    # Create app
    app = Starlette(
        routes=[
            Route("/mcp", handle_mcp_request, methods=["POST", "OPTIONS"]),
            Route("/health", health_check, methods=["GET"]),
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    return app


def _should_stream(body: Any) -> bool:
    """Check if request should stream."""
    if isinstance(body, dict):
        params = body.get("params", {})
        if isinstance(params, dict):
            args = params.get("arguments", {})
            return args.get("stream", False)
    return False


async def _handle_json(  # noqa: PLR0911
    body: Any, mcp_server: Any, validator: "SecurityValidator", origin: str
) -> JSONResponse:
    """Handle JSON-RPC requests and route to FastMCP methods.

    Implements MCP Spec 2025-06-18 JSON-RPC 2.0 protocol.

    Args:
        body: Parsed JSON-RPC request body
        mcp_server: FastMCP server instance
        validator: Security validator for CORS
        origin: Request origin for CORS headers

    Returns:
        JSONResponse with JSON-RPC 2.0 formatted result or error
    """
    request_id = body.get("id") if isinstance(body, dict) else None

    try:
        # Validate JSON-RPC structure
        if not isinstance(body, dict):
            return JSONResponse(
                JSONRPCHandler.create_error_response(
                    request_id, -32600, "Invalid Request: body must be object"
                ),
                status_code=400,
                headers=validator.get_cors_headers(origin),
            )

        method = body.get("method")
        if not method:
            return JSONResponse(
                JSONRPCHandler.create_error_response(
                    request_id, -32600, "Invalid Request: missing method"
                ),
                status_code=400,
                headers=validator.get_cors_headers(origin),
            )

        params = body.get("params", {})

        # Route to appropriate MCP method
        if method == "tools/call":
            # Extract tool name and arguments
            tool_name = params.get("name")
            if not tool_name:
                return JSONResponse(
                    JSONRPCHandler.create_error_response(
                        request_id, -32602, "Invalid params: missing tool name"
                    ),
                    status_code=400,
                    headers=validator.get_cors_headers(origin),
                )

            arguments = params.get("arguments", {})

            # Call FastMCP tool execution via public API
            # Note: FastMCP's call_tool returns the tool result directly
            result = await mcp_server.call_tool(tool_name, arguments)

            return JSONResponse(
                JSONRPCHandler.create_success_response(request_id, {"content": result}),
                headers=validator.get_cors_headers(origin),
            )

        if method == "tools/list":
            # List available tools via public API
            tools = await mcp_server.list_tools()
            # FastMCP's list_tools() already returns a list of mcp.types.Tool objects
            # These need to be serialized to dicts for JSON-RPC response
            tools_serialized = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in tools
            ]

            return JSONResponse(
                JSONRPCHandler.create_success_response(request_id, {"tools": tools_serialized}),
                headers=validator.get_cors_headers(origin),
            )

        # Unknown method
        return JSONResponse(
            JSONRPCHandler.create_error_response(request_id, -32601, f"Method not found: {method}"),
            status_code=404,
            headers=validator.get_cors_headers(origin),
        )

    except Exception:
        logger.exception("Tool execution error")
        return JSONResponse(
            JSONRPCHandler.create_error_response(request_id, -32603, "Internal error"),
            status_code=500,
            headers=validator.get_cors_headers(origin),
        )


async def _handle_streaming(
    body: Any, _mcp: Any, validator: "SecurityValidator", origin: str
) -> StreamingResponse:
    """Handle SSE streaming response."""

    async def stream_events() -> Any:
        # Placeholder for streaming implementation
        request_id = body.get("id") if isinstance(body, dict) else None
        yield f"data: {json.dumps({'jsonrpc': '2.0', 'id': request_id, 'error': {'code': -32601, 'message': 'Streaming not yet implemented'}})}\n\n"

    headers = validator.get_cors_headers(origin)
    headers.update(
        {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

    return StreamingResponse(stream_events(), media_type="text/event-stream", headers=headers)


def _cors_preflight(request: Request, validator: "SecurityValidator") -> Response:
    """Handle CORS preflight."""
    origin = request.headers.get("origin", "")
    headers = validator.get_cors_headers(origin)
    if not headers:
        return Response(status_code=403)
    return Response(status_code=200, headers=headers)


class SecurityValidator:
    """Security validation for HTTP transport."""

    def __init__(self) -> None:
        self.allowed_origins = {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        }

    def validate_origin(self, origin: str) -> bool:
        """Validate origin header."""
        if not origin:
            return False
        return origin in self.allowed_origins

    def get_cors_headers(self, origin: str) -> dict:
        """Get CORS headers if origin valid."""
        if not self.validate_origin(origin):
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept, Origin",
            "Access-Control-Max-Age": "86400",
        }

    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)


class JSONRPCHandler:
    """Handle JSON-RPC 2.0 messages."""

    @staticmethod
    def create_error_response(request_id: Any, code: int, message: str) -> dict:
        """Create error response."""
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def create_success_response(request_id: Any, result: Any) -> dict:
        """Create success response."""
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
