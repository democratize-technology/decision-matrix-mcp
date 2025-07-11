#!/usr/bin/env python3

import sys
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Add our custom imports ONE BY ONE
from decision_matrix_mcp.models import DecisionSession, Option, Criterion, Score, ModelBackend

mcp = FastMCP("decision-matrix")

@mcp.tool()
def hello(name: str) -> str:
    """Say hello"""
    return f"Hello {name}!"

class StartDecisionRequest(BaseModel):
    topic: str = Field(description="The decision topic")
    options: List[str] = Field(description="List of options")

@mcp.tool(description="Start a new decision analysis")
def start_decision_analysis(request: StartDecisionRequest) -> Dict[str, Any]:
    """Initialize a decision analysis session"""
    
    # Test using our models
    session = DecisionSession(
        session_id="test-123",
        created_at=__import__('datetime').datetime.now(__import__('datetime').UTC),
        topic=request.topic
    )
    
    return {
        "session_id": session.session_id,
        "topic": request.topic,
        "options": request.options,
        "message": f"Started analysis using DecisionSession model"
    }

if __name__ == "__main__":
    mcp.run()
