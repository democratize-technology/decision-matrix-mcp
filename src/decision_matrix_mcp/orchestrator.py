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

"""Thread orchestration for parallel criterion evaluation"""

import asyncio
import logging
import os
import re
import threading
from typing import Any

from .exceptions import (
    ChainOfThoughtError,
    ConfigurationError,
    CoTTimeoutError,
    LLMAPIError,
    LLMBackendError,
    LLMConfigurationError,
)
from .models import CriterionThread, ModelBackend, Option
from .reasoning_orchestrator import DecisionReasoningOrchestrator

# Optional dependency imports with availability flags
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"


class DecisionOrchestrator:
    """Orchestrates parallel criterion evaluation with different LLM backends"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_cot: bool = True,
        cot_timeout: float = 30.0,
    ):
        """Initialize orchestrator

        Args:
            max_retries: Maximum number of retries for failed calls
            retry_delay: Initial delay between retries (exponential backoff)
            use_cot: Whether to use Chain of Thought reasoning when available
            cot_timeout: Maximum time in seconds for CoT processing (default: 30s)
        """
        self.backends = {
            ModelBackend.BEDROCK: self._call_bedrock,
            ModelBackend.LITELLM: self._call_litellm,
            ModelBackend.OLLAMA: self._call_ollama,
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_cot = use_cot
        self.reasoning_orchestrator = DecisionReasoningOrchestrator(cot_timeout=cot_timeout)
        self._bedrock_client = None
        self._client_lock = threading.Lock()

    def _get_bedrock_client(self):
        """Get or create a Bedrock client instance (thread-safe)"""
        if self._bedrock_client is None:
            with self._client_lock:
                # Double-check pattern to avoid race conditions
                if self._bedrock_client is None:
                    session = boto3.Session()
                    region = os.environ.get(
                        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
                    )
                    self._bedrock_client = session.client("bedrock-runtime", region_name=region)
        return self._bedrock_client

    async def test_bedrock_connection(self) -> dict[str, Any]:
        """Test Bedrock connectivity and model access permissions"""
        if not BOTO3_AVAILABLE:
            return {
                "status": "error",
                "error": "boto3 not installed. Install with: pip install boto3",
                "region": "N/A",
            }

        region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        try:
            session = boto3.Session()
            bedrock = session.client("bedrock-runtime", region_name=region)

            # Test with minimal request using Haiku (cheaper and faster)
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            system_prompts = [{"text": "You are a helpful assistant."}]

            response = bedrock.converse(
                modelId=model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig={"maxTokens": 10, "temperature": 0.1},
            )

            # Parse response - converse API has cleaner structure
            response_text = ""
            if "output" in response and "message" in response["output"]:
                message_content = response["output"]["message"]["content"]
                if message_content and len(message_content) > 0:
                    response_text = message_content[0]["text"]

            return {
                "status": "ok",
                "region": region,
                "model_tested": model_id,
                "response_length": len(response_text),
                "message": "Bedrock connection successful",
                "api_version": "converse",
            }

        except (BotoCoreError, ClientError) as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "Unknown")
            error_message = str(e)

            return {
                "status": "error",
                "region": region,
                "error_code": error_code,
                "error": error_message,
                "model_tested": model_id,
                "suggestion": self._get_bedrock_error_suggestion(error_code, error_message),
            }

        except Exception as e:
            return {
                "status": "error",
                "region": region,
                "error": f"Unexpected error: {str(e)}",
                "suggestion": "Check AWS credentials and region configuration",
            }

    def _get_bedrock_error_suggestion(self, error_code: str, error_message: str) -> str:
        """Get user-friendly suggestions for common Bedrock errors"""
        if "access" in error_message.lower() or "permission" in error_message.lower():
            return (
                "Enable model access in AWS Console: Bedrock > Model access > Manage model access"
            )
        elif "region" in error_message.lower():
            return "Try us-east-1 or us-west-2 regions where Bedrock is available"
        elif "credentials" in error_message.lower():
            return (
                "Configure AWS credentials: aws configure or set AWS_PROFILE environment variable"
            )
        elif "throttling" in error_message.lower():
            return "Request rate limit exceeded. Wait a moment and try again"
        else:
            return "Check AWS Bedrock service status and model availability"

    async def evaluate_options_across_criteria(
        self, threads: dict[str, CriterionThread], options: list[Option]
    ) -> dict[str, dict[str, tuple[float | None, str]]]:
        """Evaluate all options across all criteria in parallel

        Returns:
            {criterion_name: {option_name: (score, justification)}}
        """
        all_tasks = []
        task_metadata = []

        for criterion_name, thread in threads.items():
            for option in options:
                task = self._evaluate_single_option(thread, option)
                all_tasks.append(task)
                task_metadata.append((criterion_name, option.name))

        logger.info(
            f"Starting parallel evaluation of {len(options)} options across {len(threads)} criteria"
        )
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        evaluation_results: dict[str, dict[str, tuple[float | None, str]]] = {}
        for i, result in enumerate(results):
            criterion_name, option_name = task_metadata[i]

            if criterion_name not in evaluation_results:
                evaluation_results[criterion_name] = {}

            if isinstance(result, Exception):
                logger.error(f"Error evaluating {option_name} for {criterion_name}: {result}")
                evaluation_results[criterion_name][option_name] = (None, f"Error: {str(result)}")
            elif isinstance(result, tuple):
                evaluation_results[criterion_name][option_name] = result
            else:
                logger.error(
                    f"Unexpected result type for {option_name}/{criterion_name}: {type(result)}"
                )
                evaluation_results[criterion_name][option_name] = (
                    None,
                    "Error: Unexpected result type",
                )

        return evaluation_results

    async def _evaluate_single_option(
        self, thread: CriterionThread, option: Option
    ) -> tuple[float | None, str]:
        """Evaluate a single option with a single criterion

        Returns:
            (score, justification) where score is None if abstained
        """
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
                        thread, option, bedrock_client
                    )
                except CoTTimeoutError:
                    # Fallback to standard evaluation on timeout
                    logger.warning(
                        f"CoT timeout for {option.name} on {thread.criterion.name}, falling back to standard evaluation"
                    )
                    response = await self._get_thread_response(thread)
                    thread.add_message("assistant", response)
                    return self._parse_evaluation_response(response)
                except ChainOfThoughtError as e:
                    # Log CoT-specific errors and fallback
                    logger.error(f"CoT error for {option.name}: {e}")
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
            else:
                # Standard evaluation without CoT
                response = await self._get_thread_response(thread)
                thread.add_message("assistant", response)
                return self._parse_evaluation_response(response)

        except LLMBackendError as e:
            logger.error(
                f"LLM backend error evaluating {option.name} for {thread.criterion.name}: {e}"
            )
            return (None, e.user_message)
        except Exception:
            logger.exception(
                f"Unexpected error evaluating {option.name} for {thread.criterion.name}"
            )
            return (None, "Evaluation failed due to an unexpected error")

    def _parse_evaluation_response(self, response: str) -> tuple[float | None, str]:
        """Parse the structured evaluation response with multiple fallback patterns

        Expected format:
        SCORE: [number or NO_RESPONSE]
        JUSTIFICATION: [text]

        Fallback patterns:
        - Score: X/10
        - Rating: X
        - [NO_RESPONSE] anywhere in response
        """
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
                # Extract justification even for abstentions
                justification = self._extract_justification(normalized_response)
                return (None, justification)

            # Extract score with multiple patterns
            score = self._extract_score(normalized_response)

            # Extract justification
            justification = self._extract_justification(normalized_response)

            # Validate that we have at least a score or justification
            if score is None and justification == "No justification provided":
                logger.warning(
                    f"Could not parse meaningful content from response: {response[:200]}..."
                )
                return (None, "Could not parse evaluation from response")

            return (score, justification)

        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            # Return partial parse if possible
            try:
                justification = self._extract_justification(response)
                return (None, justification)
            except Exception:
                return (None, "Parse error: Unable to extract evaluation")

    def _extract_score(self, response: str) -> float | None:
        """Extract numeric score using multiple patterns"""
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

    def _extract_justification(self, response: str) -> str:
        """Extract justification using multiple patterns"""
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
                    r"\s*(SCORE:|Rating:|Score\s*=)", "", justification, flags=re.IGNORECASE
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

    async def _get_thread_response(self, thread: CriterionThread) -> str:
        """Get response from a single thread with retry logic"""
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
            except Exception as e:
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
                    logger.error(f"Non-retryable error for {thread.criterion.name}: {e}")
                    raise e

                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {thread.criterion.name}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retries failed for {thread.criterion.name}: {e}")

        raise last_error or Exception("Unknown error in thread response")

    async def _call_bedrock(self, thread: CriterionThread) -> str:
        """Call AWS Bedrock for criterion evaluation"""
        if not BOTO3_AVAILABLE:
            raise LLMConfigurationError(
                backend="bedrock",
                message="boto3 is not installed. Please install with: pip install boto3",
            )

        try:
            session = boto3.Session()
            # Get region from environment or default to us-east-1
            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            bedrock = session.client("bedrock-runtime", region_name=region)

            # Prepare messages for converse API format
            messages = []
            for msg in thread.conversation_history:
                # Converse API uses a simpler format
                messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

            # Choose model
            model_id = thread.criterion.model_name or "anthropic.claude-3-sonnet-20240229-v1:0"

            # Prepare inference configuration
            inference_config = {
                "maxTokens": thread.criterion.max_tokens,
                "temperature": thread.criterion.temperature,
            }

            # Prepare system prompts
            system_prompts = [{"text": thread.criterion.system_prompt}]

            # Prepare kwargs for converse API
            converse_kwargs = {
                "modelId": model_id,
                "messages": messages,
                "system": system_prompts,
                "inferenceConfig": inference_config,
            }

            response = bedrock.converse(**converse_kwargs)

            if "output" in response and "message" in response["output"]:
                message_content = response["output"]["message"]["content"]
                if message_content and len(message_content) > 0:
                    return message_content[0]["text"]

            raise LLMAPIError(
                backend="bedrock",
                message=f"Invalid response format from Bedrock converse API: {response}",
                user_message="Unexpected response format from LLM",
            ) from None

        except ImportError as e:
            raise LLMConfigurationError(
                backend="bedrock",
                message=f"boto3 dependency missing: {e}",
                original_error=e,
            ) from e
        except (BotoCoreError, ClientError) as e:
            # Enhanced error logging for diagnostics
            error_message = str(e)
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "Unknown")

            logger.error("Bedrock API error details:")
            logger.error(f"  - Error Code: {error_code}")
            logger.error(f"  - Error Message: {error_message}")
            logger.error(f"  - Model ID: {model_id}")
            logger.error(f"  - Region: {region}")
            logger.error("  - Using converse API with inference config")

            # Check for specific error types
            if "rate limit" in error_message.lower() or "throttling" in error_message.lower():
                user_message = "Request rate limit exceeded, please try again later"
            elif "invalid" in error_message.lower() and "model" in error_message.lower():
                user_message = f"Invalid model ID: {model_id}"
            elif "access" in error_message.lower() or "permission" in error_message.lower():
                user_message = f"No access to model {model_id} in region {region}. Check Bedrock model access in AWS Console."
            elif "region" in error_message.lower():
                user_message = (
                    f"Bedrock not available in region {region}. Try us-east-1 or us-west-2."
                )
            else:
                user_message = "LLM service temporarily unavailable"

            raise LLMAPIError(
                backend="bedrock",
                message=f"Bedrock API call failed: {e}",
                user_message=user_message,
                original_error=e,
            ) from e
        except Exception as e:
            # Only catch truly unexpected errors
            if isinstance(e, (LLMBackendError, ConfigurationError)):
                raise  # Re-raise our custom exceptions
            raise LLMBackendError(
                backend="bedrock",
                message=f"Unexpected error in Bedrock call: {e}",
                user_message="An unexpected error occurred",
                original_error=e,
            ) from e

    async def _call_litellm(self, thread: CriterionThread) -> str:
        """Call LiteLLM for criterion evaluation"""
        if not LITELLM_AVAILABLE:
            raise LLMConfigurationError(
                backend="litellm",
                message="litellm is not installed. Please install with: pip install litellm",
            )

        try:
            # Prepare messages
            messages = [{"role": "system", "content": thread.criterion.system_prompt}]
            for msg in thread.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Choose model
            model = thread.criterion.model_name or "gpt-3.5-turbo"

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=thread.criterion.temperature,
                max_tokens=thread.criterion.max_tokens,
            )

            return response.choices[0].message.content

        except ImportError as e:
            raise LLMConfigurationError(
                backend="litellm",
                message=f"litellm dependency missing: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            # Check for specific error types
            error_message = str(e)
            if "rate limit" in error_message.lower() or "quota" in error_message.lower():
                user_message = "API rate limit exceeded, please try again later"
            elif "api key" in error_message.lower() or "authentication" in error_message.lower():
                user_message = "API authentication failed, check your API key"
            elif "model" in error_message.lower() and "not found" in error_message.lower():
                user_message = f"Model not available: {model}"
            elif isinstance(e, (LLMBackendError, ConfigurationError)):
                raise  # Re-raise our custom exceptions
            else:
                user_message = "LLM service temporarily unavailable"

            raise LLMAPIError(
                backend="litellm",
                message=f"LiteLLM API call failed: {e}",
                user_message=user_message,
                original_error=e,
            ) from e

    async def _call_ollama(self, thread: CriterionThread) -> str:
        """Call Ollama for criterion evaluation"""
        if not HTTPX_AVAILABLE:
            raise LLMConfigurationError(
                backend="ollama",
                message="httpx is not installed. Please install with: pip install httpx",
            )

        try:
            # Prepare messages
            messages = [{"role": "system", "content": thread.criterion.system_prompt}]
            for msg in thread.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Choose model
            model = thread.criterion.model_name or "llama2"

            async with httpx.AsyncClient(timeout=60.0) as client:
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                # Build options with criterion parameters
                options = {"temperature": thread.criterion.temperature, "num_ctx": 4096}

                response = await client.post(
                    f"{ollama_host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": options,
                    },
                )

                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = f"Ollama API error: {error_data['error']}"
                    except Exception:
                        pass

                    if response.status_code == 404:
                        raise LLMAPIError(
                            backend="ollama",
                            message=error_msg,
                            user_message=f"Model not available in Ollama: {model}",
                        )
                    elif response.status_code == 503:
                        raise LLMAPIError(
                            backend="ollama",
                            message=error_msg,
                            user_message="Ollama service is not running",
                        )
                    else:
                        raise LLMAPIError(
                            backend="ollama",
                            message=error_msg,
                            user_message="Ollama service temporarily unavailable",
                        )

                result = response.json()
                return result["message"]["content"]

        except ImportError as e:
            raise LLMConfigurationError(
                backend="ollama",
                message=f"httpx dependency missing: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            if isinstance(e, (LLMBackendError, ConfigurationError)):
                raise  # Re-raise our custom exceptions

            # Check for connection errors
            error_message = str(e)
            if "connection" in error_message.lower() or "refused" in error_message.lower():
                user_message = "Cannot connect to Ollama service. Is it running?"
            else:
                user_message = "Ollama service error"

            raise LLMAPIError(
                backend="ollama",
                message=f"Ollama call failed: {e}",
                user_message=user_message,
                original_error=e,
            ) from e
