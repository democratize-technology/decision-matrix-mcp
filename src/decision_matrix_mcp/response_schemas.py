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

"""JSON schemas and validators for structured LLM response parsing."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# JSON Schema for evaluation responses
EVALUATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "oneOf": [
                {"type": "number", "minimum": 0, "maximum": 10},
                {"type": "null"},
                {"type": "string", "enum": ["NO_RESPONSE", "N/A", "ABSTAIN"]},
            ],
        },
        "reasoning": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "status": {"type": "string", "enum": ["success", "abstain", "error"]},
    },
    "required": ["score", "reasoning", "status"],
    "additionalProperties": False,
}


class EvaluationResponse:
    """Structured evaluation response with validation."""

    def __init__(
        self,
        score: float | str | None = None,
        reasoning: str = "",
        confidence: float | None = None,
        status: str = "success",
    ) -> None:
        self.score = score
        self.reasoning = reasoning
        self.confidence = confidence
        self.status = status

        # Validate and normalize
        self._validate_and_normalize()

    def _validate_and_normalize(self) -> None:
        """Validate and normalize response fields."""
        # Handle abstention patterns
        if isinstance(self.score, str):
            abstention_keywords = ["NO_RESPONSE", "N/A", "ABSTAIN"]
            if any(keyword in str(self.score).upper() for keyword in abstention_keywords):
                self.score = None
                self.status = "abstain"

        # Validate score range and clamp if needed
        if isinstance(self.score, (int, float)):
            self.score = max(1.0, min(10.0, float(self.score)))

        # Ensure reasoning exists
        if not self.reasoning.strip():
            self.reasoning = "No reasoning provided"

        # Validate confidence
        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResponse":
        """Create from dictionary with validation."""
        return cls(
            score=data.get("score"),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence"),
            status=data.get("status", "success"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationResponse":
        """Create from JSON string with validation."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON format: {e}")

    def is_abstention(self) -> bool:
        """Check if this is an abstention response."""
        return self.score is None or self.status == "abstain"

    def get_legacy_format(self) -> tuple[float | None, str]:
        """Get response in legacy (score, justification) format for compatibility."""
        return (self.score, self.reasoning)


def generate_json_prompt_suffix() -> str:
    """Generate prompt suffix requesting JSON format response."""
    return """

Please respond with a valid JSON object in this exact format:
{
    "score": 7.5,
    "reasoning": "Detailed explanation of your evaluation...",
    "confidence": 0.8,
    "status": "success"
}

Guidelines:
- score: Number between 1-10, or null if you cannot evaluate (use null for abstention)
- reasoning: Detailed explanation of your evaluation
- confidence: Your confidence level (0-1), optional
- status: "success", "abstain", or "error"

For abstention, use: {"score": null, "reasoning": "Explanation of why you cannot evaluate", "status": "abstain"}

IMPORTANT: Respond with ONLY the JSON object, no additional text before or after."""


def parse_structured_response(response: str) -> EvaluationResponse:
    """Parse structured JSON response with intelligent fallback.

    Args:
        response: Raw LLM response string

    Returns:
        EvaluationResponse object

    Raises:
        ValueError: If response cannot be parsed in any format
    """
    # Clean response
    response = response.strip()

    # Try JSON parsing first
    try:
        return _parse_json_response(response)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.debug(f"JSON parsing failed: {e}, falling back to regex parsing")
        # Fall back to legacy regex parsing
        return _parse_legacy_response(response)


def _parse_json_response(response: str) -> EvaluationResponse:
    """Parse JSON response format."""
    # Try to extract JSON from response (handle cases where there's extra text)
    json_str = _extract_json_from_response(response)

    if not json_str:
        raise ValueError("No JSON found in response")

    try:
        data = json.loads(json_str)
        return EvaluationResponse.from_dict(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def _extract_json_from_response(response: str) -> str | None:
    """Extract JSON object from response text."""
    # Look for JSON object boundaries
    start_idx = response.find("{")
    if start_idx == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    end_idx = -1

    for i in range(start_idx, len(response)):
        if response[i] == "{":
            brace_count += 1
        elif response[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        return None

    return response[start_idx:end_idx]


def _parse_legacy_response(response: str) -> EvaluationResponse:
    """Parse legacy text response format using existing regex logic."""
    # Use legacy regex parsing logic directly to avoid circular imports
    score, reasoning = _parse_legacy_regex(response)

    # Convert to structured format
    status = "abstain" if score is None else "success"

    return EvaluationResponse(score=score, reasoning=reasoning, status=status)


def _parse_legacy_regex(response: str) -> tuple[float | None, str]:
    """Legacy regex parsing logic extracted to avoid circular imports."""
    try:
        # Normalize response for consistent parsing
        normalized_response = response.strip()

        # Check for abstention patterns first
        abstention_patterns = [
            "[NO_RESPONSE]",
            "NO_RESPONSE",
            "not applicable",
            "cannot evaluate",
            "unable to score",
            "abstain",
        ]

        if any(pattern.lower() in normalized_response.lower() for pattern in abstention_patterns):
            # Extract justification even for abstentions
            justification = _extract_justification_legacy(normalized_response)
            return (None, justification)

        # Extract score with multiple patterns
        score = _extract_score_legacy(normalized_response)

        # Extract justification
        justification = _extract_justification_legacy(normalized_response)

        # Validate that we have at least a score or justification
        if score is None and justification == "No justification provided":
            logger.warning(f"Could not parse meaningful content from response: {response[:200]}...")
            return (None, "Could not parse evaluation from response")

        return (score, justification)

    except Exception as e:
        logger.exception(f"Error parsing evaluation response: {e}")
        # Return partial parse if possible
        try:
            justification = _extract_justification_legacy(response)
            return (None, justification)
        except Exception:
            return (None, "Parse error: Unable to extract evaluation")


def _extract_score_legacy(response: str) -> float | None:
    """Legacy score extraction logic."""
    import re

    # Pattern 1: SCORE: X
    score_patterns = [
        (r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", 1),
        (r"SCORE:\s*([0-9]+)/10", 1),
        (r"SCORE:.*?([0-9]+(?:\.[0-9]+)?)", 1),  # SCORE: with text before number
        (r"Rating:\s*([0-9]+(?:\.[0-9]+)?)", 1),
        (r"([0-9]+(?:\.[0-9]+)?)/10", 1),
        (r"Score\s*=\s*([0-9]+(?:\.[0-9]+)?)", 1),
        (r"score\s+is\s+([0-9]+(?:\.[0-9]+)?)", 1),  # "score is X"
        (r"rate\s+this\s+([0-9]+(?:\.[0-9]+)?)", 1),  # "rate this X"
        (r"^([0-9]+(?:\.[0-9]+)?)\s*$", 1),  # Just a number
    ]

    for pattern, group in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                score = float(match.group(group))
                # Clamp to 1-10 range
                return max(1.0, min(10.0, score))
            except (ValueError, IndexError):
                continue

    return None


def _extract_justification_legacy(response: str) -> str:
    """Legacy justification extraction logic."""
    import re

    # Pattern 1: JUSTIFICATION: text
    patterns = [
        r"JUSTIFICATION:\s*(.+)",
        r"Justification:\s*(.+)",
        r"Reasoning:\s*(.+)",
        r"Explanation:\s*(.+)",
        r"Because\s+(.+)",
        r"Rationale:\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            justification = match.group(1).strip()
            # Clean up common endings
            justification = re.sub(
                r"\s*(SCORE:|Rating:|Score\s*=)",
                "",
                justification,
                flags=re.IGNORECASE,
            )
            return justification.strip()

    # Fallback: If response has multiple lines, take everything after first line
    lines = response.strip().split("\n")
    if len(lines) > 1:
        # Skip first line if it contains score
        if re.search(r"(score|rating|^\d+)", lines[0], re.IGNORECASE):
            return "\n".join(lines[1:]).strip()

    # Last fallback: Return trimmed response if it's descriptive
    if len(response.strip()) > 20 and not re.match(r"^\d+(\.\d+)?$", response.strip()):
        return response.strip()[:500]  # Cap length

    return "No justification provided"


def validate_response_schema(data: dict[str, Any]) -> bool:
    """Validate response against JSON schema."""
    try:
        # Basic validation - would use jsonschema library in production
        required_fields = ["score", "reasoning", "status"]
        for field in required_fields:
            if field not in data:
                return False

        # Score validation
        score = data["score"]
        if score is not None and not isinstance(score, (int, float)):
            if not isinstance(score, str) or score.upper() not in ["NO_RESPONSE", "N/A", "ABSTAIN"]:
                return False

        # Status validation
        return data["status"] in ["success", "abstain", "error"]
    except Exception:
        return False
