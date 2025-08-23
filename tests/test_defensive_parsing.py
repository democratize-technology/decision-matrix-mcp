"""
Comprehensive tests for defensive code patterns in response parsing.

These tests specifically target defensive programming branches that are often
missed in coverage reports but critical for production reliability.
"""

from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from decision_matrix_mcp.models import Criterion, CriterionThread, ModelBackend, Option
from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestDefensiveParsing:
    """Test defensive parsing patterns in orchestrator response handling."""

    @pytest.fixture()
    def orchestrator(self):
        """Create orchestrator with mocked backend for testing."""
        return DecisionOrchestrator()

    @pytest.mark.parametrize(
        ("malformed_response", "expected_score", "expected_contains"),
        [
            # Empty responses
            ("", None, "Could not parse evaluation"),
            ("   ", None, "Could not parse evaluation"),
            ("\n\n\t\t", None, "Could not parse evaluation"),
            # Invalid score formats
            ("SCORE: invalid\nJUSTIFICATION: test", None, "test"),
            ("SCORE: NaN\nJUSTIFICATION: not a number", None, "not a number"),
            ("SCORE: infinity\nJUSTIFICATION: infinite", None, "infinite"),
            ("SCORE: -infinity\nJUSTIFICATION: negative infinite", None, "negative infinite"),
            # Score boundary clamping - defensive pattern testing
            ("SCORE: 15\nJUSTIFICATION: too high", 10.0, "too high"),  # Clamp to max
            ("SCORE: -5\nJUSTIFICATION: too low", 1.0, "too low"),  # Clamp to min
            ("SCORE: 0\nJUSTIFICATION: zero", 1.0, "zero"),  # Zero clamped to 1
            ("SCORE: 0.5\nJUSTIFICATION: half", 1.0, "half"),  # Below 1 clamped
            # No structure responses
            ("Random text with no format", None, "Random text with no format"),
            ("Just a sentence.", None, "Just a sentence."),
            ("12345", None, "No justification provided"),  # Just a number
            # Abstention patterns - defensive handling
            ("[NO_RESPONSE]", None, "No justification provided"),
            ("NO_RESPONSE", None, "No justification provided"),
            ("not applicable to this option", None, "not applicable to this option"),
            ("cannot evaluate this criterion", None, "cannot evaluate this criterion"),
            ("unable to score", None, "unable to score"),
            ("abstain from rating", None, "abstain from rating"),
            # Edge cases with valid scores but defensive fallbacks
            ("SCORE: 5.5", 5.5, "No justification provided"),  # Score only
            (
                "JUSTIFICATION: Only justification provided",
                None,
                "Only justification provided",
            ),  # No score
            # Malformed but parseable
            ("Score is 8 out of 10\nBecause it's good", 8.0, "it's good"),
            ("Rating: 7.5/10\nExcellent choice", 7.5, "Excellent choice"),
            # Very long content - defensive truncation
            ("SCORE: 6\nJUSTIFICATION: " + "x" * 1000, 6.0, "x" * 500),  # Should truncate
            # Unicode and special characters
            ("SCORE: 8\nJUSTIFICATION: Très bien! 🎉", 8.0, "Très bien! 🎉"),
        ],
    )
    def test_parse_evaluation_defensive_branches(
        self, orchestrator, malformed_response, expected_score, expected_contains
    ):
        """Test defensive parsing handles all edge cases gracefully."""
        score, justification = orchestrator._parse_evaluation_response_legacy(malformed_response)

        assert score == expected_score, f"Score mismatch for input: {malformed_response[:50]}..."
        assert (
            expected_contains in justification
        ), f"Justification should contain '{expected_contains}', got: {justification}"

    @pytest.mark.parametrize(
        ("response", "expected_score", "expected_justification_contains"),
        [
            # Test structured response fallback to legacy parsing
            (
                '{"score": "invalid"}',
                "invalid",
                "No reasoning provided",
            ),  # Invalid JSON structure handled
            ('{"malformed json"', None, "malformed json"),  # Malformed JSON fallback to text
            ("null", None, "null"),  # Null response handling
        ],
    )
    def test_structured_to_legacy_fallback(
        self, orchestrator, response, expected_score, expected_justification_contains
    ):
        """Test that structured parsing failures are handled gracefully."""
        score, justification = orchestrator._parse_evaluation_response(response)

        # Should handle malformed structured responses gracefully
        assert score == expected_score
        assert expected_justification_contains in justification

    def test_parsing_with_regex_edge_cases(self, orchestrator):
        """Test regex patterns with edge cases that could cause ReDoS or failures."""
        edge_cases = [
            "SCORE: 5.5.5.5\nJUSTIFICATION: Multiple decimals",  # Malformed decimal
            "SCORE: 5.\nJUSTIFICATION: Trailing decimal",  # Trailing decimal
            "SCORE: .5\nJUSTIFICATION: Leading decimal",  # Leading decimal
            "SCORE: 5e10\nJUSTIFICATION: Scientific notation",  # Scientific notation
            "SCORE: +5\nJUSTIFICATION: Plus sign",  # Plus prefix
            "SCORE:5\nJUSTIFICATION: No space",  # No space after colon
            "SCORE : 5\nJUSTIFICATION: Extra space",  # Extra space
            "score: 5\nJUSTIFICATION: Lowercase",  # Lowercase
            "SCORE: 5 \nJUSTIFICATION: Trailing space",  # Trailing space on score
        ]

        for response in edge_cases:
            score, justification = orchestrator._parse_evaluation_response_legacy(response)
            # Should not crash and should produce some result
            assert score is None or (1.0 <= score <= 10.0), f"Invalid score for: {response}"
            assert len(justification) > 0, f"Empty justification for: {response}"

    @pytest.mark.asyncio()
    async def test_evaluation_with_backend_timeout_defensive_handling(self, orchestrator):
        """Test defensive handling when backend times out during evaluation."""
        criterion = Criterion(
            name="test_criterion",
            description="Test",
            model_backend=ModelBackend.BEDROCK,
            weight=1.0,
        )
        thread = CriterionThread(id="test", criterion=criterion)
        option = Option(name="TestOption", description="Test option")

        # Mock backend to timeout
        with patch.object(orchestrator.backend_factory, "create_backend") as mock_create:
            mock_backend = AsyncMock()
            mock_backend.generate_response.side_effect = TimeoutError("Backend timeout")
            mock_create.return_value = mock_backend

            # Should handle timeout gracefully with defensive response
            score, justification = await orchestrator._evaluate_single_option(thread, option)

            assert score is None
            assert "timeout" in justification.lower() or "error" in justification.lower()

    @pytest.mark.asyncio()
    async def test_chain_of_thought_fallback_defensive_pattern(self, orchestrator):
        """Test Chain of Thought fallback to standard evaluation when CoT fails."""
        criterion = Criterion(
            name="test_criterion",
            description="Test",
            model_backend=ModelBackend.BEDROCK,
            weight=1.0,
        )
        thread = CriterionThread(id="test", criterion=criterion)
        option = Option(name="TestOption", description="Test option")

        # Enable CoT
        orchestrator.use_cot = True

        # Mock CoT to fail, standard evaluation to succeed
        with (
            patch.object(
                type(orchestrator.reasoning_orchestrator), "is_available", new_callable=PropertyMock
            ) as mock_available,
            patch.object(
                orchestrator.reasoning_orchestrator, "evaluate_with_reasoning"
            ) as mock_cot,
            patch.object(
                orchestrator,
                "_get_thread_response",
                return_value="SCORE: 8\nJUSTIFICATION: Standard evaluation worked",
            ),
            patch.object(orchestrator, "_get_bedrock_client"),
        ):
            mock_available.return_value = True
            from decision_matrix_mcp.exceptions import CoTTimeoutError

            mock_cot.side_effect = CoTTimeoutError("CoT timeout")

            score, justification = await orchestrator._evaluate_single_option(thread, option)

            # Should fall back to standard evaluation
            assert score == 8.0
            assert "Standard evaluation worked" in justification

    def test_extract_score_boundary_conditions(self, orchestrator):
        """Test score extraction with boundary conditions and defensive limits."""
        test_cases = [
            ("SCORE: 10.0", 10.0),  # Max valid
            ("SCORE: 1.0", 1.0),  # Min valid
            ("SCORE: 10.1", 10.0),  # Clamp to max
            ("SCORE: 0.9", 1.0),  # Clamp to min
            ("SCORE: 999", 10.0),  # Large number clamped
            ("SCORE: -999", 1.0),  # Negative clamped
        ]

        for response, expected_score in test_cases:
            score = orchestrator._extract_score(response)
            assert (
                score == expected_score
            ), f"Failed for {response}: got {score}, expected {expected_score}"

    def test_extract_justification_defensive_patterns(self, orchestrator):
        """Test justification extraction with defensive patterns."""
        test_cases = [
            # Normal cases
            ("JUSTIFICATION: This is good", "This is good"),
            ("Reasoning: Well thought out", "Well thought out"),
            ("Explanation: Clear logic", "Clear logic"),
            # Defensive fallbacks
            ("No clear justification pattern", "No clear justification pattern"),
            ("", "No justification provided"),
            ("SCORE: 5", "No justification provided"),  # Only score
            # Cleanup patterns
            ("JUSTIFICATION: Good choice SCORE: 8", "Good choice"),  # Remove trailing score
            ("JUSTIFICATION: Excellent\nSCORE: 9", "Excellent"),  # Remove trailing score on newline
        ]

        for response, expected in test_cases:
            result = orchestrator._extract_justification(response)
            assert (
                expected in result
            ), f"Failed for '{response}': got '{result}', expected to contain '{expected}'"
