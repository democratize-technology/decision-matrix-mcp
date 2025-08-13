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

"""Chain of Thought integration for structured decision reasoning."""

import asyncio
import logging
from typing import Any

try:
    from chain_of_thought import TOOL_SPECS, AsyncChainOfThoughtProcessor

    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

from .exceptions import ChainOfThoughtError, CoTProcessingError, CoTTimeoutError
from .models import CriterionThread, Option

logger = logging.getLogger(__name__)


class DecisionReasoningOrchestrator:
    """Orchestrates Chain of Thought reasoning for decision evaluation."""

    def __init__(self, cot_timeout: float | None = None) -> None:
        """Initialize the reasoning orchestrator.

        Args:
            cot_timeout: Maximum time in seconds for CoT processing.
                        If None, uses configuration default.
        """
        # Import config here to avoid circular imports
        from .config import config

        # Use configuration value if not explicitly provided
        resolved_timeout = (
            cot_timeout if cot_timeout is not None else config.performance.cot_timeout_seconds
        )
        if not COT_AVAILABLE:
            logger.warning(
                "chain-of-thought-tool not available. Install with: pip install chain-of-thought-tool",
            )
        self.cot_timeout = resolved_timeout
        self._cot_available = COT_AVAILABLE

    @property
    def is_available(self) -> bool:
        """Check if Chain of Thought reasoning is available."""
        return self._cot_available

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text to prevent injection attacks.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Remove control characters and normalize whitespace
        import re
        import unicodedata

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Limit length to prevent resource exhaustion
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "... (truncated)"

        # Remove potential prompt injection patterns
        injection_patterns = [
            r"(?i)ignore\s+previous\s+instructions",
            r"(?i)disregard\s+all\s+prior",
            r'(?i)system\s*:\s*["\']',
            r'(?i)assistant\s*:\s*["\']',
            r"(?i)<\|.*?\|>",  # Special tokens
        ]

        for pattern in injection_patterns:
            text = re.sub(pattern, "[FILTERED]", text)

        return text.strip()

    def get_cot_tools(self) -> list[dict[str, Any]]:
        """Get Chain of Thought tool specifications for Bedrock."""
        if not COT_AVAILABLE:
            return []
        return TOOL_SPECS

    async def evaluate_with_reasoning(
        self,
        thread: CriterionThread,
        option: Option,
        bedrock_client: Any,
    ) -> tuple[float | None, str, dict[str, Any]]:
        """Evaluate an option using structured Chain of Thought reasoning.

        Returns:
            Tuple of (score, justification, reasoning_summary)
        """
        if not COT_AVAILABLE:
            # Fallback to simple evaluation
            return None, "Chain of Thought not available", {}

        # Validate and sanitize option data
        option_name = self._sanitize_input(option.name)
        option_description = self._sanitize_input(
            option.description or "No additional description provided",
        )

        # Create a CoT processor for this evaluation with unique ID
        import uuid

        unique_id = uuid.uuid4().hex[:8]
        processor = AsyncChainOfThoughtProcessor(
            conversation_id=f"{thread.id}-{option_name}-{unique_id}",
        )

        # Prepare the enhanced system prompt
        enhanced_system_prompt = f"""{thread.criterion.system_prompt}

You have access to Chain of Thought tools to structure your reasoning:
- Use chain_of_thought_step to work through your evaluation systematically
- Start with Problem Definition stage to understand what you're evaluating
- Move through Analysis to examine the option against the criterion
- End with Conclusion stage to provide your final score
- Set next_step_needed=false when you're ready to give the final score

After your reasoning is complete, provide your evaluation in this format:
SCORE: [number 1-10 or NO_RESPONSE]
JUSTIFICATION: [summary of your reasoning]"""

        # Prepare initial request with CoT tools
        messages = []

        # Add conversation history
        for msg in thread.conversation_history:
            messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

        # Add the evaluation prompt with sanitized inputs
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": f"""Evaluate this option: {option_name}

Option Description: {option_description}

Use chain_of_thought_step to reason through your evaluation systematically.
Remember to follow the scoring format at the end:
SCORE: [1-10 or NO_RESPONSE if not applicable]
JUSTIFICATION: [your reasoning summary]""",
                    },
                ],
            },
        )

        request = {
            "modelId": thread.criterion.model_name or "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": messages,
            "system": [{"text": enhanced_system_prompt}],
            "toolConfig": {"tools": self.get_cot_tools()},
            "inferenceConfig": {
                "temperature": thread.criterion.temperature,
                "maxTokens": thread.criterion.max_tokens,
            },
        }

        try:
            # Process the tool loop with CoT and timeout
            result = await asyncio.wait_for(
                processor.process_tool_loop(
                    bedrock_client=bedrock_client,
                    initial_request=request,
                    max_iterations=15,  # Allow sufficient reasoning steps
                ),
                timeout=self.cot_timeout,
            )

            # Extract the final response
            final_text = ""
            if "output" in result and "message" in result["output"]:
                content = result["output"]["message"].get("content", [])
                for item in content:
                    if "text" in item:
                        final_text += item["text"] + "\n"

            # Get reasoning summary with timeout
            try:
                reasoning_summary = await asyncio.wait_for(
                    processor.get_reasoning_summary(),
                    timeout=config.performance.cot_summary_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout getting reasoning summary")
                reasoning_summary = {}

            # Parse score and justification from final response
            score, justification = self._parse_evaluation_response(final_text)

            return score, justification, reasoning_summary

        except asyncio.TimeoutError as e:
            logger.exception("CoT evaluation timed out after %ss", self.cot_timeout)
            raise CoTTimeoutError(self.cot_timeout) from e
        except ChainOfThoughtError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error("Unexpected error in CoT evaluation: %s", e, exc_info=True)
            raise CoTProcessingError(
                f"Failed to process reasoning: {e!s}",
                stage="evaluation",
            ) from e
        finally:
            # Always clean up the processor
            try:
                processor.clear_reasoning()
            except Exception as cleanup_error:
                logger.warning("Error during processor cleanup: %s", cleanup_error)

    def _parse_evaluation_response(self, response: str) -> tuple[float | None, str]:
        """Parse the evaluation response to extract score and justification.

        Raises:
            CoTProcessingError: If response format is invalid
        """
        import re

        if not response or not isinstance(response, str):
            raise CoTProcessingError(
                "Invalid response format: expected non-empty string",
                stage="response_parsing",
            )

        # Look for SCORE: pattern
        score_match = re.search(
            r"SCORE:\s*([0-9]+(?:\.[0-9]+)?|NO_RESPONSE)",
            response,
            re.IGNORECASE,
        )
        score = None
        if score_match:
            score_text = score_match.group(1)
            if score_text.upper() != "NO_RESPONSE":
                try:
                    score = float(score_text)
                    if score < 1.0 or score > 10.0:
                        logger.warning("Score %s outside valid range, clamping to 1-10", score)
                    score = max(1.0, min(10.0, score))  # Clamp to 1-10
                except ValueError as e:
                    logger.exception("Failed to parse score '%s'", score_text)
                    raise CoTProcessingError(
                        f"Invalid score format: {score_text}",
                        stage="score_parsing",
                    ) from e

        # Look for JUSTIFICATION: pattern
        just_match = re.search(r"JUSTIFICATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        justification = "No justification provided"
        if just_match:
            justification = just_match.group(1).strip()
            # Clean up any trailing patterns
            justification = re.sub(r"\s*(SCORE:|$)", "", justification, flags=re.IGNORECASE).strip()

        # Validate we got at least something useful
        if score is None and "NO_RESPONSE" not in response.upper():
            logger.warning("Response missing valid score: %s...", response[:200])

        return score, justification
