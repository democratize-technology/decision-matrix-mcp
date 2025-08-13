"""Tests for robust response parsing in orchestrator"""

import pytest

from decision_matrix_mcp.orchestrator import DecisionOrchestrator


class TestResponseParser:
    """Test the robust response parser with various edge cases"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return DecisionOrchestrator()

    def test_standard_format(self, orchestrator):
        """Test parsing standard SCORE/JUSTIFICATION format"""
        response = """SCORE: 8.5
JUSTIFICATION: This option provides excellent performance with minimal overhead."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 8.5
        assert justification == "This option provides excellent performance with minimal overhead."

    def test_score_out_of_range(self, orchestrator):
        """Test score clamping to 1-10 range"""
        # Test above range
        response = "SCORE: 15\nJUSTIFICATION: Over the top!"
        score, _ = orchestrator._parse_evaluation_response(response)
        assert score == 10.0

        # Test below range
        response = "SCORE: 0.5\nJUSTIFICATION: Too low!"
        score, _ = orchestrator._parse_evaluation_response(response)
        assert score == 1.0

    def test_abstention_patterns(self, orchestrator):
        """Test various abstention patterns"""
        abstention_responses = [
            "SCORE: [NO_RESPONSE]\nJUSTIFICATION: Not applicable to this criterion",
            "NO_RESPONSE - This criterion cannot be evaluated",
            "Unable to score this option based on the given criterion",
            "I cannot evaluate this option",
            "This is not applicable",
            "I must abstain from scoring this",
        ]

        for response in abstention_responses:
            score, justification = orchestrator._parse_evaluation_response(response)
            assert score is None
            assert len(justification) > 0

    def test_alternative_score_formats(self, orchestrator):
        """Test alternative score formats"""
        test_cases = [
            ("Score: 7/10", 7.0),
            ("Rating: 8.2", 8.2),
            ("Score = 9", 9.0),
            ("8.5/10", 8.5),
            ("6", 6.0),  # Just a number
        ]

        for response, expected in test_cases:
            score, _ = orchestrator._parse_evaluation_response(response)
            assert score == expected

    def test_alternative_justification_formats(self, orchestrator):
        """Test alternative justification formats"""
        test_cases = [
            ("SCORE: 7\nReasoning: Good performance", "Good performance"),
            ("SCORE: 8\nExplanation: Works well", "Works well"),
            ("SCORE: 9\nBecause it handles edge cases", "it handles edge cases"),
            ("SCORE: 7\nRationale: Simple and effective", "Simple and effective"),
        ]

        for response, expected in test_cases:
            _, justification = orchestrator._parse_evaluation_response(response)
            assert justification == expected

    def test_multiline_responses(self, orchestrator):
        """Test parsing multiline responses"""
        response = """8.5
This is a strong option because:
- It handles edge cases well
- Performance is excellent
- Easy to maintain"""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 8.5
        assert "This is a strong option because:" in justification
        assert "- It handles edge cases well" in justification

    def test_malformed_responses(self, orchestrator):
        """Test graceful handling of malformed responses"""
        malformed_responses = [
            "",  # Empty
            "   ",  # Whitespace only
            "Not a valid response format",  # No score
            "SCORE: abc\nJUSTIFICATION: Bad score",  # Non-numeric score
        ]

        for response in malformed_responses:
            score, justification = orchestrator._parse_evaluation_response(response)
            # Should handle gracefully without exceptions
            assert score is None or isinstance(score, float)
            assert isinstance(justification, str)

    def test_mixed_case_parsing(self, orchestrator):
        """Test case-insensitive parsing"""
        response = """score: 7.5
justification: Good option with minor drawbacks"""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 7.5
        assert justification == "Good option with minor drawbacks"

    def test_extra_whitespace(self, orchestrator):
        """Test handling of extra whitespace"""
        response = """

        SCORE:    8

        JUSTIFICATION:    Handles the requirements well

        """

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 8.0
        assert justification == "Handles the requirements well"

    def test_score_in_justification(self, orchestrator):
        """Test that scores in justification don't interfere"""
        response = """SCORE: 7
JUSTIFICATION: This scores an 8 on usability but only 6 on performance, averaging to 7."""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 7.0
        assert "This scores an 8 on usability" in justification

    def test_partial_parse_on_error(self, orchestrator):
        """Test that parser attempts partial parse on errors"""
        response = "This option is excellent for the use case"

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score is None
        assert justification == response  # Should return the response as justification

    def test_unicode_handling(self, orchestrator):
        """Test parsing with unicode characters"""
        response = """SCORE: 8.5
JUSTIFICATION: This option provides 💯 performance with ✨ features"""

        score, justification = orchestrator._parse_evaluation_response(response)
        assert score == 8.5
        assert "💯" in justification
        assert "✨" in justification

    def test_extract_score_edge_cases(self, orchestrator):
        """Test _extract_score method directly"""
        test_cases = [
            ("The score is 7.5 out of 10", 7.5),
            ("I rate this 9/10", 9.0),
            ("Score: N/A", None),
            ("SCORE: -", None),
        ]

        for response, expected in test_cases:
            score = orchestrator._extract_score(response)
            assert score == expected

    def test_extract_justification_edge_cases(self, orchestrator):
        """Test _extract_justification method directly"""
        # Test justification extraction without score prefix
        response = "This is a comprehensive analysis of the option"
        justification = orchestrator._extract_justification(response)
        assert justification == response

        # Test empty response
        justification = orchestrator._extract_justification("")
        assert justification == "No justification provided"

        # Test very long justification (should be capped)
        long_response = "x" * 1000
        justification = orchestrator._extract_justification(long_response)
        assert len(justification) == 500
