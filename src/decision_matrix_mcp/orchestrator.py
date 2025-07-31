"""Thread orchestration for parallel criterion evaluation"""

import asyncio
import json
import logging
import os
import re

from .exceptions import ConfigurationError, LLMBackendError
from .models import CriterionThread, ModelBackend, Option

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"


class DecisionOrchestrator:
    """Orchestrates parallel criterion evaluation with different LLM backends"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize orchestrator

        Args:
            max_retries: Maximum number of retries for failed calls
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.backends = {
            ModelBackend.BEDROCK: self._call_bedrock,
            ModelBackend.LITELLM: self._call_litellm,
            ModelBackend.OLLAMA: self._call_ollama,
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

        evaluation_results = {}
        for i, result in enumerate(results):
            criterion_name, option_name = task_metadata[i]

            if criterion_name not in evaluation_results:
                evaluation_results[criterion_name] = {}

            if isinstance(result, Exception):
                logger.error(f"Error evaluating {option_name} for {criterion_name}: {result}")
                evaluation_results[criterion_name][option_name] = (None, f"Error: {str(result)}")
            else:
                evaluation_results[criterion_name][option_name] = result

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
        """Parse the structured evaluation response

        Expected format:
        SCORE: [number or NO_RESPONSE]
        JUSTIFICATION: [text]
        """
        try:
            # Extract score
            score_match = re.search(r"SCORE:\s*([^\n]+)", response, re.IGNORECASE)
            if not score_match:
                return (None, f"Could not parse score from response: {response[:200]}...")

            score_text = score_match.group(1).strip()

            # Check for abstention
            if NO_RESPONSE in score_text.upper() or "NO_RESPONSE" in score_text:
                score = None
            else:
                # Try to extract numeric score
                score_num_match = re.search(r"(\d+(?:\.\d+)?)", score_text)
                if score_num_match:
                    score = float(score_num_match.group(1))
                    # Clamp to 1-10 range
                    score = max(1.0, min(10.0, score))
                else:
                    score = None

            # Extract justification
            justification_match = re.search(
                r"JUSTIFICATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL
            )
            if justification_match:
                justification = justification_match.group(1).strip()
            else:
                justification = "No justification provided"

            return (score, justification)

        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return (None, f"Parse error: {str(e)}")

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
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError

            # Initialize Bedrock client
            session = boto3.Session()
            # Get region from environment or default to us-east-1
            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            bedrock = session.client("bedrock-runtime", region_name=region)

            # Prepare messages for Bedrock format
            messages = []
            for msg in thread.conversation_history:
                # Bedrock requires content to be a list of content blocks
                messages.append(
                    {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
                )

            # Choose model
            model_id = thread.criterion.model_name or "anthropic.claude-3-sonnet-20240229-v1:0"

            # Call Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.1,  # Low temperature for consistent scoring
                "system": thread.criterion.system_prompt,
                "messages": messages,
            }

            response = bedrock.invoke_model(modelId=model_id, body=json.dumps(request_body))

            # Parse response - Bedrock returns content as a list
            response_body = json.loads(response["body"].read())
            if "content" in response_body and len(response_body["content"]) > 0:
                return response_body["content"][0]["text"]
            else:
                raise Exception("Invalid response format from Bedrock")

        except ImportError:
            raise Exception("boto3 not available for Bedrock backend") from None
        except (BotoCoreError, ClientError) as e:
            raise Exception(f"Bedrock API error: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Bedrock call failed: {str(e)}") from e

    async def _call_litellm(self, thread: CriterionThread) -> str:
        """Call LiteLLM for criterion evaluation"""
        try:
            import litellm

            # Prepare messages
            messages = [{"role": "system", "content": thread.criterion.system_prompt}]
            for msg in thread.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Choose model
            model = thread.criterion.model_name or "gpt-3.5-turbo"

            # Call LiteLLM
            response = await litellm.acompletion(
                model=model, messages=messages, temperature=0.1, max_tokens=1024
            )

            return response.choices[0].message.content

        except ImportError:
            raise Exception("litellm not available") from None
        except Exception as e:
            raise Exception(f"LiteLLM call failed: {str(e)}") from e

    async def _call_ollama(self, thread: CriterionThread) -> str:
        """Call Ollama for criterion evaluation"""
        try:
            import httpx

            # Prepare messages
            messages = [{"role": "system", "content": thread.criterion.system_prompt}]
            for msg in thread.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Choose model
            model = thread.criterion.model_name or "llama2"

            # Call Ollama API
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Get Ollama host from environment or default
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                response = await client.post(
                    f"{ollama_host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_ctx": 4096},
                    },
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")

                result = response.json()
                return result["message"]["content"]

        except ImportError:
            raise Exception("httpx not available for Ollama backend") from None
        except Exception as e:
            raise Exception(f"Ollama call failed: {str(e)}") from e
