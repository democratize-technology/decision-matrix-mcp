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
import re
from typing import Any
import unicodedata
import uuid

try:
    from chain_of_thought import TOOL_SPECS, AsyncChainOfThoughtProcessor  # type: ignore[import]

    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

from .config import config
from .convergence import ConvergenceManager, create_evaluation_id
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

        # Initialize convergence manager for score stabilization
        self.convergence_manager = ConvergenceManager(
            convergence_threshold=0.1,  # Variance threshold for convergence
            min_samples=3,  # Need at least 3 scores to check convergence
            window_size=5,  # Track up to 5 recent scores
        )

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
        """Evaluate an option using structured Chain of Thought reasoning with convergence detection.

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

        # Create evaluation ID for convergence tracking
        evaluation_id = create_evaluation_id(thread.criterion.name, option_name)

        # Create a CoT processor for this evaluation with unique ID
        unique_id = uuid.uuid4().hex[:8]
        processor = AsyncChainOfThoughtProcessor(
            conversation_id=f"{thread.id}-{option_name}-{unique_id}",
        )

        # Prepare the enhanced system prompt with convergence instructions
        enhanced_system_prompt = f"""{thread.criterion.system_prompt}

You have access to Chain of Thought tools to structure your reasoning:
- Use chain_of_thought_step to work through your evaluation systematically
- Start with Problem Definition stage to understand what you're evaluating
- Move through Analysis to examine the option against the criterion
- Include intermediate score estimates during Analysis stages (format: "Current score estimate: X/10")
- End with Conclusion stage to provide your final score
- Set next_step_needed=false when you're ready to give the final score

After your reasoning is complete, provide your evaluation in this format:
SCORE: [number 1-10 or NO_RESPONSE]
JUSTIFICATION: [summary of your reasoning]"""

        # Prepare initial request with CoT tools
        messages = []

        # Add conversation history, filtering out system messages (AWS Bedrock compliance)
        filtered_history = [
            msg for msg in thread.conversation_history if msg.get("role") != "system"
        ]
        messages = [
            {"role": msg["role"], "content": [{"text": msg["content"]}]} for msg in filtered_history
        ]

        # Add the evaluation prompt with sanitized inputs
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": f"""Evaluate this option: {option_name}

Option Description: {option_description}

Use chain_of_thought_step to reason through your evaluation systematically.
Include intermediate score estimates during your analysis (format: "Current score estimate: X/10").
Remember to follow the scoring format at the end:
SCORE: [1-10 or NO_RESPONSE if not applicable]
JUSTIFICATION: [your reasoning summary]""",
                    },
                ],
            },
        )

        # AWS Bedrock requires conversations to start with a user message
        # Since we always append a user message above, we need to check if the first message from
        # conversation history isn't user and insert a default user message if needed
        if len(messages) > 1 and messages[0]["role"] != "user":
            # Insert a default user message at the beginning
            messages.insert(
                0, {"role": "user", "content": [{"text": "Please help me evaluate this option."}]}
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
            # Use convergence-aware iterative evaluation
            result = await asyncio.wait_for(
                self._process_with_convergence(
                    processor=processor,
                    bedrock_client=bedrock_client,
                    initial_request=request,
                    evaluation_id=evaluation_id,
                    max_iterations=15,
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

            # Add convergence metrics to reasoning summary
            convergence_metrics = self.convergence_manager.get_tracker(
                evaluation_id,
            ).get_convergence_metrics()
            reasoning_summary["convergence"] = convergence_metrics

            # Parse score and justification from final response
            score, justification = self._parse_evaluation_response(final_text)

            # Clean up convergence tracker for this evaluation
            self.convergence_manager.cleanup_tracker(evaluation_id)

        except asyncio.TimeoutError as e:
            logger.exception("CoT evaluation timed out after %ss", self.cot_timeout)
            self.convergence_manager.cleanup_tracker(evaluation_id)
            raise CoTTimeoutError(self.cot_timeout) from e
        except ChainOfThoughtError:
            # Re-raise our custom exceptions
            self.convergence_manager.cleanup_tracker(evaluation_id)
            raise
        except Exception as e:
            logger.exception("Unexpected error in CoT evaluation")
            self.convergence_manager.cleanup_tracker(evaluation_id)
            raise CoTProcessingError(
                f"Failed to process reasoning: {e}",
                stage="evaluation",
            ) from e
        else:
            return score, justification, reasoning_summary
        finally:
            # Always clean up the processor
            try:
                processor.clear_reasoning()
            except (AttributeError, RuntimeError) as cleanup_error:
                logger.warning("Error during processor cleanup: %s", cleanup_error)

    async def _process_with_convergence(
        self,
        processor: Any,
        bedrock_client: Any,
        initial_request: dict[str, Any],
        evaluation_id: str,
        max_iterations: int = 15,
    ) -> dict[str, Any]:
        """Process Chain of Thought with convergence detection.

        Args:
            processor: AsyncChainOfThoughtProcessor instance
            bedrock_client: Bedrock client for API calls
            initial_request: Initial request parameters
            evaluation_id: Unique identifier for this evaluation
            max_iterations: Maximum number of reasoning iterations

        Returns:
            Final result from Chain of Thought processing
        """
        logger.info("Starting convergence-aware evaluation for %s", evaluation_id)

        # Initialize tracking variables
        current_request = initial_request.copy()
        iteration = 0
        convergence_tracker = self.convergence_manager.get_tracker(evaluation_id)

        while iteration < max_iterations:
            iteration += 1

            try:
                # Execute single Chain of Thought iteration
                response = await bedrock_client.converse(**current_request)

                # Extract response content
                response_text = ""
                if "output" in response and "message" in response["output"]:
                    content = response["output"]["message"].get("content", [])
                    for item in content:
                        if "text" in item:
                            response_text += item["text"] + "\n"

                # Look for intermediate score estimates in the response
                intermediate_score = self._extract_intermediate_score(response_text)
                if intermediate_score is not None:
                    logger.debug(
                        "Iteration %s: Found intermediate score %s",
                        iteration,
                        intermediate_score,
                    )

                    # Track convergence
                    converged = convergence_tracker.add_score(intermediate_score)
                    if converged:
                        logger.info(
                            "🎯 Score convergence detected for %s at iteration %s! Early termination triggered.",
                            evaluation_id,
                            iteration,
                        )
                        # Continue with final response generation
                        break

                # Check if Chain of Thought indicates completion
                if "next_step_needed" in response_text.lower() and "false" in response_text.lower():
                    logger.debug("Chain of Thought completion detected at iteration %s", iteration)
                    break

                # Handle tool use continuation
                if "output" in response and "message" in response["output"]:
                    output_message = response["output"]["message"]

                    # Check for tool use
                    if "content" in output_message:
                        for content_item in output_message["content"]:
                            if "toolUse" in content_item:
                                # Process tool use with Chain of Thought processor
                                try:
                                    tool_result = await processor.handle_tool_use(
                                        bedrock_client,
                                        current_request,
                                        response,
                                    )
                                    if tool_result:
                                        current_request = tool_result.get(
                                            "next_request",
                                            current_request,
                                        )
                                        continue
                                except (
                                    ChainOfThoughtError,
                                    CoTProcessingError,
                                    asyncio.TimeoutError,
                                ) as tool_error:
                                    logger.warning("Tool processing error: %s", tool_error)
                                    break

                # If no tool use, we're done
                if not any(
                    "toolUse" in item
                    for item in output_message.get("content", [])
                    if isinstance(item, dict)
                ):
                    break

            except Exception:
                logger.exception("Error in iteration %s", iteration)
                if iteration == 1:
                    # If first iteration fails, re-raise
                    raise
                # Otherwise, use the last successful response
                break

        # Log final convergence status
        final_metrics = convergence_tracker.get_convergence_metrics()
        logger.info(
            "Evaluation %s completed after %s iterations. Convergence: %s, Final variance: %.4f",
            evaluation_id,
            iteration,
            final_metrics.get("converged", False),
            final_metrics.get("current_variance", 0.0),
        )

        # Return the final response in expected format
        return {"output": {"message": response["output"]["message"]}}

    def _extract_intermediate_score(self, response_text: str) -> float | None:
        """Extract intermediate score estimate from Chain of Thought response.

        Args:
            response_text: Text response from Chain of Thought step

        Returns:
            Extracted score (1-10) or None if no score found
        """
        if not response_text:
            return None

        # Look for patterns like "Current score estimate: 7/10" or "Score estimate: 8.5"
        patterns = [
            r"current\s+score\s+estimate:\s*([0-9]+(?:\.[0-9]+)?)",
            r"score\s+estimate:\s*([0-9]+(?:\.[0-9]+)?)",
            r"estimated\s+score:\s*([0-9]+(?:\.[0-9]+)?)",
            r"tentative\s+score:\s*([0-9]+(?:\.[0-9]+)?)",
            r"preliminary\s+score:\s*([0-9]+(?:\.[0-9]+)?)",
            r"interim\s+score:\s*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10",  # X/10 format
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to 1-10 range
                    return max(1.0, min(10.0, score))
                except (ValueError, IndexError):
                    continue

        return None

    def _parse_evaluation_response(self, response: str) -> tuple[float | None, str]:
        """Parse the evaluation response to extract score and justification.

        Raises:
            CoTProcessingError: If response format is invalid
        """
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
