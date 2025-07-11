#!/usr/bin/env python3
"""
Minimal MCP server - tracing execution step by step
"""

import sys
import traceback

print("🔍 STEP 1: Starting imports", file=sys.stderr)

try:
    from mcp.server.fastmcp import FastMCP
    print("🔍 STEP 2: FastMCP imported successfully", file=sys.stderr)
except Exception as e:
    print(f"❌ STEP 2 FAILED: FastMCP import error: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    exit(1)

print("🔍 STEP 3: Creating FastMCP server", file=sys.stderr)

try:
    mcp = FastMCP("trace-test")
    print("🔍 STEP 4: FastMCP server created successfully", file=sys.stderr)
except Exception as e:
    print(f"❌ STEP 4 FAILED: Server creation error: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    exit(1)

print("🔍 STEP 5: Registering test tool", file=sys.stderr)

try:
    @mcp.tool()
    def test_tool(message: str) -> str:
        """A test tool"""
        return f"Test response: {message}"
    
    print("🔍 STEP 6: Tool registered successfully", file=sys.stderr)
except Exception as e:
    print(f"❌ STEP 6 FAILED: Tool registration error: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    exit(1)

print("🔍 STEP 7: About to call mcp.run()", file=sys.stderr)

# Add a handler to see what happens when the process exits
import atexit

def exit_handler():
    print("🔍 PROCESS EXITING: atexit handler called", file=sys.stderr)

atexit.register(exit_handler)

try:
    print("🔍 STEP 8: Calling mcp.run() with default transport", file=sys.stderr)
    mcp.run()
    print("🔍 STEP 9: mcp.run() returned (this should not happen for stdio)", file=sys.stderr)
except KeyboardInterrupt:
    print("🔍 STEP 9: KeyboardInterrupt received", file=sys.stderr)
except SystemExit as e:
    print(f"🔍 STEP 9: SystemExit called with code: {e.code}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    exit(e.code)
except Exception as e:
    print(f"❌ STEP 9 FAILED: mcp.run() error: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    exit(1)

print("🔍 STEP 10: Script ended (should not reach here)", file=sys.stderr)
