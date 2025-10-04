#!/usr/bin/env python3
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

"""Entry point for the Decision Matrix MCP server."""

import asyncio
import logging
import os
import sys

# Configure logging to stderr to avoid interfering with MCP protocol
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the MCP server with stdio transport (default)."""
    logger.info("Starting Decision Matrix MCP server (stdio)...")
    try:
        from . import main as run_server

        run_server()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("Server error")
        raise


def http_main(host: str = "127.0.0.1", port: int = 8081) -> None:
    """Run the MCP server with HTTP transport using FastMCP's native support.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 8081)
    """
    logger.info("Starting Decision Matrix MCP server (HTTP) on %s:%s", host, port)
    try:
        from . import create_mcp_server, initialize_server_components

        # Initialize server components
        try:
            initialize_server_components()
            logger.info("Server components initialized")
        except Exception:
            logger.exception("Failed to initialize server")
            sys.exit(1)

        # Create server with HTTP settings
        mcp = create_mcp_server(host=host, port=port)
        # Use run_streamable_http_async for HTTP transport
        asyncio.run(mcp.run_streamable_http_async())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("Server error")
        raise


if __name__ == "__main__":
    # Check if HTTP mode requested
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8081"))
        http_main(host=host, port=port)
    else:
        main()
