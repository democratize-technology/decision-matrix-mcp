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

"""Thread orchestration for parallel criterion evaluation."""

import asyncio
import logging
import os
import re
import threading
from typing import Any

from .backends import BackendFactory
from .config import config
from .exceptions import (
    ChainOfThoughtError,
    ConfigurationError,
    CoTTimeoutError,
    LLMBackendError,
    LLMConfigurationError,
)
from .models import Criterion, CriterionThread, ModelBackend, Option
from .reasoning_orchestrator import DecisionReasoningOrchestrator
from .response_schemas import parse_structured_response

# Optional dependency imports for legacy compatibility
try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Use importlib to check availability without importing
import importlib.util

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None
HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None

logger = logging.getLogger(__name__)

NO_RESPONSE = "[NO_RESPONSE]"


class DecisionOrchestrator:
    """Orchestrates parallel evaluation of decision options across multiple criteria.

    Manages the execution of LLM-based evaluations using configurable backends
    (Bedrock, LiteLLM, Ollama) with retry logic and graceful error handling.
    """

    def __init__(
        self,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        use_cot: bool = True,
        cot_timeout: float | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        # Initialize backend factory
        self.backend_factory = backend_factory or BackendFactory()

        # Legacy backend mapping for existing facade methods
        self.backends = {
            ModelBackend.BEDROCK: self._call_bedrock,
            ModelBackend.LITELLM: self._call_litellm,
            ModelBackend.OLLAMA: self._call_ollama,
        }

        # Use configuration values with optional overrides
        self.max_retries = (
            max_retries if max_retries is not None else config.performance.max_retries
        )
        self.retry_delay = (
            retry_delay if retry_delay is not None else config.performance.retry_delay_seconds
        )
        self.use_cot = use_cot

        # Initialize reasoning orchestrator with configurable timeout
        cot_timeout_resolved = (
            cot_timeout if cot_timeout is not None else config.performance.cot_timeout_seconds
        )
        self.reasoning_orchestrator = DecisionReasoningOrchestrator(
            cot_timeout=cot_timeout_resolved,
        )

        # Legacy Bedrock client for CoT compatibility
        self._bedrock_client = None
        self._client_lock = threading.Lock()

    def _get_bedrock_client(self) -> Any:
        if self._bedrock_client is None:
            with self._client_lock:
                # Double-check pattern to avoid race conditions
                if self._bedrock_client is None:
                    session = boto3.Session()
                    region = os.environ.get(
                        "AWS_REGION",
                        os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                    )
                    self._bedrock_client = session.client("bedrock-runtime", region_name=region)
        return self._bedrock_client

    async def test_bedrock_connection(self) -> dict[str, Any]:
        """Test Bedrock backend connectivity - delegates to backend."""
        # Check if backend is available first
        if not self.backend_factory.validate_backend_availability(ModelBackend.BEDROCK):
            return {
                "status": "error",
                "error": "boto3 not installed. Install with: pip install boto3",
                "region": "N/A",
            }

        # Use the bedrock backend for connection testing
        try:
            backend = self.backend_factory.create_backend(ModelBackend.BEDROCK)

            # Test with minimal request using Haiku (cheaper and faster)
            test_criterion = Criterion(
                name="test",
                description="test",
                model_name="anthropic.claude-3-haiku-20240307-v1:0",
                temperature=0.1,
                max_tokens=10,
            )
            test_thread = CriterionThread(id="test", criterion=test_criterion)
            test_thread.add_message("user", "Hi")

            # Make test call
            response_text = await backend.generate_response(test_thread)

            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

            return {
                "status": "ok",
                "region": region,
                "model_tested": test_criterion.model_name,
                "response_length": len(response_text),
                "message": "Bedrock connection successful",
                "api_version": "converse",
            }

        except Exception as e:  # noqa: BLE001
            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

            # Extract error details if it's a Bedrock error
            error_code = "Unknown"
            error_message = str(e)

            if hasattr(e, "original_error") and BOTO3_AVAILABLE:
                original = e.original_error
                if hasattr(original, "response"):
                    error_code = original.response.get("Error", {}).get("Code", "Unknown")
                    error_message = str(original)

            return {
                "status": "error",
                "region": region,
                "error_code": error_code,
                "error": error_message,
                "model_tested": "anthropic.claude-3-haiku-20240307-v1:0",
                "suggestion": self._get_bedrock_error_suggestion(error_code, error_message),
            }

    def _get_bedrock_error_suggestion(
        self,
        error_code: str,  # noqa: ARG002
        error_message: str,
    ) -> str:
        if "access" in error_message.lower() or "permission" in error_message.lower():
            return (
                "Enable model access in AWS Console: Bedrock > Model access > Manage model access"
            )
        if "region" in error_message.lower():
            return "Try us-east-1 or us-west-2 regions where Bedrock is available"
        if "credentials" in error_message.lower():
            return (
                "Configure AWS credentials: aws configure or set AWS_PROFILE environment variable"
            )
        if "throttling" in error_message.lower():
            return "Request rate limit exceeded. Wait a moment and try again"
        return "Check AWS Bedrock service status and model availability"

    async def evaluate_options_across_criteria(
        self,
        threads: dict[str, CriterionThread],
        options: list[Option],
    ) -> dict[str, dict[str, tuple[float | None, str]]]:
        """Evaluate all options against all criteria in parallel.

        Returns dict mapping criterion_name -> option_name -> (score, reasoning).
        """
        all_tasks = []
        task_metadata = []

        for criterion_name, thread in threads.items():
            for option in options:
                task = self._evaluate_single_option(thread, option)
                all_tasks.append(task)
                task_metadata.append((criterion_name, option.name))

        logger.info(
            "Starting parallel evaluation of %d options across %d criteria",
            len(options),
            len(threads),
        )
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        evaluation_results: dict[str, dict[str, tuple[float | None, str]]] = {}
        for i, result in enumerate(results):
            criterion_name, option_name = task_metadata[i]

            if criterion_name not in evaluation_results:
                evaluation_results[criterion_name] = {}

            if isinstance(result, Exception):
                logger.error("Error evaluating %s for %s: %s", option_name, criterion_name, result)
                evaluation_results[criterion_name][option_name] = (None, f"Error: {result!s}")
            elif isinstance(result, tuple):
                evaluation_results[criterion_name][option_name] = result
            else:
                logger.error(
                    "Unexpected result type for %s/%s: %s",
                    option_name,
                    criterion_name,
                    type(result),
                )
                evaluation_results[criterion_name][option_name] = (
                    None,
                    "Error: Unexpected result type",
                )

        return evaluation_results

    async def _evaluate_single_option(
        self,
        thread: CriterionThread,
        option: Option,
    ) -> tuple[float | None, str]:
        prompt = f"""Evaluate this option: {option.name}

Option Description: {option.description or "No additional description provided"}

Remember to follow the scoring format:
SCORE: [1-10 or NO_RESPONSE if not applicable]
JUSTIFICATION: [your reasoning]"""

        thread.add_message("user", prompt)

        try:
            # Use Chain of Thought for Bedrock if enabled
            if (
                self.use_cot
                and thread.criterion.model_backend == ModelBackend.BEDROCK
                and self.reasoning_orchestrator.is_available
            ):
                # Get shared Bedrock client for CoT evaluation
                bedrock_client = self._get_bedrock_client()

                # Evaluate with structured reasoning
                try:
                    (
                        score,
                        justification,
                        reasoning_summary,
                    ) = await self.reasoning_orchestrator.evaluate_with_reasoning(
                        thread,
                        option,
                        bedrock_client,
                    )
                except CoTTimeoutError:
                    # Fallback to standard evaluation on timeout
                    logger.warning(
                        "CoT timeout for %s on %s, falling back to standard evaluation",
                        option.name,
                        thread.criterion.name,
                    )
                    response = await self._get_thread_response(thread)
                    thread.add_message("assistant", response)
                    return self._parse_evaluation_response(response)
                except ChainOfThoughtError:
                    # Log CoT-specific errors and fallback
                    logger.exception("CoT error for %s", option.name)
                    response = await self._get_thread_response(thread)
                    thread.add_message("assistant", response)
                    return self._parse_evaluation_response(response)

                # Add reasoning summary to thread if available
                if reasoning_summary and "total_steps" in reasoning_summary:
                    thread.add_message(
                        "assistant",
                        f"[Reasoned through {reasoning_summary['total_steps']} steps across stages: {', '.join(reasoning_summary.get('stages_covered', []))}]",
                    )

                # Add the final response to thread
                response = f"SCORE: {score if score is not None else 'NO_RESPONSE'}\nJUSTIFICATION: {justification}"
                thread.add_message("assistant", response)

                return (score, justification)
            # Standard evaluation without CoT
            response = await self._get_thread_response(thread)
            thread.add_message("assistant", response)
            return self._parse_evaluation_response(response)

        except LLMBackendError as e:
            logger.exception(
                "LLM backend error evaluating %s for %s",
                option.name,
                thread.criterion.name,
            )
            return (None, e.user_message)
        except Exception:
            logger.exception(
                "Unexpected error evaluating %s for %s",
                option.name,
                thread.criterion.name,
            )
            return (None, "Evaluation failed due to an unexpected error")

    def _parse_evaluation_response(self, response: str) -> tuple[float | None, str]:
        """Parse evaluation response using structured JSON with regex fallback.

        This method first attempts to parse the response as structured JSON,
        then falls back to the legacy regex-based parsing for compatibility.

        Args:
            response: Raw LLM response string

        Returns:
            Tuple of (score, justification) - score is None for abstentions
        """
        try:
            # Use structured parser with intelligent fallback
            structured_response = parse_structured_response(response)
            return structured_response.get_legacy_format()

        except Exception as e:  # noqa: BLE001
            logger.warning("Structured parsing failed, using legacy fallback: %s", e)
            # Fall back to legacy regex parsing
            return self._parse_evaluation_response_legacy(response)

    def _parse_evaluation_response_legacy(  # noqa: PLR0911
        self, response: str
    ) -> tuple[float | None, str]:
        """Legacy regex-based response parsing for backward compatibility."""
        try:
            # Normalize response for consistent parsing
            normalized_response = response.strip()

            # Check for abstention patterns first
            abstention_patterns = [
                NO_RESPONSE,
                "NO_RESPONSE",
                "not applicable",
                "cannot evaluate",
                "unable to score",
                "abstain",
            ]

            if any(
                pattern.lower() in normalized_response.lower() for pattern in abstention_patterns
            ):
                # For abstentions, try to extract justification, but fall back to original text if no formal pattern found
                justification = self._extract_justification(normalized_response)
                if justification == "No justification provided":
                    # Special case: formal abstention tokens should return "No justification provided"
                    formal_abstentions = ["[NO_RESPONSE]", "NO_RESPONSE"]
                    if any(
                        formal.lower() == normalized_response.strip().lower()
                        for formal in formal_abstentions
                    ):
                        justification = "No justification provided"
                    else:
                        justification = normalized_response[
                            :500
                        ]  # Use original text for informal abstentions
                return (None, justification)

            # Extract score with multiple patterns
            score = self._extract_score(normalized_response)

            # Extract justification
            justification = self._extract_justification(normalized_response)

            # Enhanced validation: handle descriptive responses without scores
            if score is None and justification == "No justification provided":
                # Special case: pure numbers should be treated as abstention with "No justification provided"
                if re.match(r"^[+-]?\d+(\.\d+)?$", normalized_response.strip()):
                    return (None, "No justification provided")

                # For other descriptive responses without formal structure, use the original text
                if len(normalized_response) > 3:
                    return (
                        None,
                        normalized_response[:500],
                    )  # Return original text capped at 500 chars

                logger.warning(
                    "Could not parse meaningful content from response: %s...",
                    response[:200],
                )
                return (None, "Could not parse evaluation from response")

        except Exception:
            logger.exception("Error parsing evaluation response")
            # Return partial parse if possible
            try:
                justification = self._extract_justification(response)
            except Exception:  # noqa: BLE001
                return (None, "Parse error: Unable to extract evaluation")
            else:
                return (None, justification)
        else:
            return (score, justification)

    def _extract_score(self, response: str) -> float | None:
        # Pattern 1: SCORE: X (now handles negative numbers correctly)
        score_patterns = [
            (r"SCORE:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", 1),  # Fixed: handles negative numbers
            (r"SCORE:\s*([+-]?[0-9]+)/10", 1),  # Fixed: handles negative numbers
            (r"Rating:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", 1),  # Fixed: handles negative numbers
            (r"([+-]?[0-9]+(?:\.[0-9]+)?)/10", 1),  # Fixed: handles negative numbers
            (r"Score\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?)", 1),  # Fixed: handles negative numbers
            (r"score\s+is\s+([+-]?[0-9]+(?:\.[0-9]+)?)", 1),  # "score is X" - fixed
            (r"rate\s+this\s+([+-]?[0-9]+(?:\.[0-9]+)?)", 1),  # "rate this X" - fixed
            # Removed pure number pattern - numbers without context should not be treated as scores
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

    def _extract_justification(self, response: str) -> str:
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
        if len(lines) > 1 and re.search(r"(score|rating|^\d+)", lines[0], re.IGNORECASE):
            # Skip first line if it contains score
            return "\n".join(lines[1:]).strip()

        # Last fallback: Return trimmed response if it's descriptive
        if len(response.strip()) > 20 and not re.match(r"^\d+(\.\d+)?$", response.strip()):
            return response.strip()[:500]  # Cap length

        return "No justification provided"

    async def _get_thread_response(self, thread: CriterionThread) -> str:
        backend_fn = self.backends.get(thread.criterion.model_backend)
        if not backend_fn:
            raise ConfigurationError(
                f"Unknown model backend: {thread.criterion.model_backend}",
                f"Model backend '{thread.criterion.model_backend}' is not configured",
            )

        # Try with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await backend_fn(thread)
            except Exception as e:  # noqa: PERF203
                last_error = e
                error_str = str(e).lower()

                # Don't retry on non-transient errors
                if any(
                    term in error_str
                    for term in [
                        "api_key",
                        "credentials",
                        "not found",
                        "invalid",
                        "unauthorized",
                        "forbidden",
                        "model not found",
                    ]
                ):
                    logger.exception("Non-retryable error for %s", thread.criterion.name)
                    raise

                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        "Attempt %d failed for %s: %s. Retrying in %ss...",
                        attempt + 1,
                        thread.criterion.name,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception("All retries failed for %s", thread.criterion.name)

        raise last_error or Exception("Unknown error in thread response")

    async def _call_bedrock(self, thread: CriterionThread) -> str:
        """Legacy facade method - delegates to BedrockBackend."""
        backend = self.backend_factory.create_backend(ModelBackend.BEDROCK)
        return await backend.generate_response(thread)

    async def _call_litellm(self, thread: CriterionThread) -> str:
        """Legacy facade method - delegates to LiteLLMBackend."""
        if not LITELLM_AVAILABLE:
            raise LLMConfigurationError(
                backend="litellm",
                message="litellm is not installed. Please install with: pip install litellm",
            )

        backend = self.backend_factory.create_backend(ModelBackend.LITELLM)
        return await backend.generate_response(thread)

    async def _call_ollama(self, thread: CriterionThread) -> str:
        """Legacy facade method - delegates to OllamaBackend."""
        if not HTTPX_AVAILABLE:
            raise LLMConfigurationError(
                backend="ollama",
                message="httpx is not installed. Please install with: pip install httpx",
            )

        backend = self.backend_factory.create_backend(ModelBackend.OLLAMA)
        return await backend.generate_response(thread)

    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        # Clean up backend factory instances
        self.backend_factory.cleanup()

        # Legacy Bedrock client cleanup
        if self._bedrock_client:
            with self._client_lock:
                self._bedrock_client = None
                logger.info("Legacy Bedrock client cleaned up")
