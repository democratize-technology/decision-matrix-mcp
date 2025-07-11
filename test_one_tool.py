#!/usr/bin/env python3
"""
Decision Matrix MCP Server - Debugging version with one tool
"""

import sys
import logging
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import our modules
from decision_matrix_mcp.models import DecisionSession, Option, Criterion, Score, ModelBackend
from decision_matrix_mcp.orchestrator import DecisionOrchestrator
from decision_matrix_mcp.session_manager import session_manager, SessionValidator

# Configure logging to stderr only
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

print("🔍 [DEBUG] Starting full Decision Matrix MCP server...", file=sys.stderr)

# Initialize MCP server
mcp = FastMCP("decision-matrix")

# Create shared orchestrator instance
orchestrator = DecisionOrchestrator()

print("🔍 [DEBUG] Basic setup complete", file=sys.stderr)

# Just one simple tool to start
class StartDecisionAnalysisRequest(BaseModel):
    topic: str = Field(description="The decision topic")
    options: List[str] = Field(description="List of options")

@mcp.tool(
    description="Start a new decision analysis session"
)
def start_decision_analysis(request: StartDecisionAnalysisRequest) -> Dict[str, Any]:
    """Initialize a decision analysis session"""
    try:
        print(f"🔍 [DEBUG] Processing start_decision_analysis: {request.topic}", file=sys.stderr)
        
        # Create new session using our session manager
        session = session_manager.create_session(
            topic=request.topic,
            initial_options=request.options
        )
        
        return {
            "session_id": session.session_id,
            "topic": request.topic,
            "options": request.options,
            "message": f"Decision analysis initialized with {len(request.options)} options"
        }
        
    except Exception as e:
        print(f"❌ [ERROR] Error in start_decision_analysis: {e}", file=sys.stderr)
        return {"error": f"Failed to create session: {str(e)}"}

print("🔍 [DEBUG] Tool registered successfully", file=sys.stderr)

def main():
    """Run the Decision Matrix MCP server"""
    try:
        print("🔍 [DEBUG] Starting server...", file=sys.stderr)
        mcp.run()
        print("🔍 [DEBUG] Server exited", file=sys.stderr)
    except KeyboardInterrupt:
        print("🔍 [DEBUG] Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"❌ [ERROR] Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    print("🔍 [DEBUG] Main execution starting", file=sys.stderr)
    main()
