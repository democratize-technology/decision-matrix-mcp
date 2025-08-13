"""Tests for structured JSON response parsing with fallback to regex"""

import json
import pytest

from decision_matrix_mcp.response_schemas import (
    EvaluationResponse,
    parse_structured_response,
    generate_json_prompt_suffix,
    validate_response_schema,
    _parse_json_response,
    _parse_legacy_response,
    _extract_json_from_response,
)


class TestEvaluationResponse:
    """Test the EvaluationResponse class"""

    def test_basic_creation(self):
        """Test basic response creation"""
        response = EvaluationResponse(
            score=7.5, reasoning="Good performance overall", confidence=0.8, status="success"
        )

        assert response.score == 7.5
        assert response.reasoning == "Good performance overall"
        assert response.confidence == 0.8
        assert response.status == "success"
        assert not response.is_abstention()

    def test_abstention_creation(self):
        """Test abstention response creation"""
        response = EvaluationResponse(
            score=None, reasoning="Cannot evaluate this criterion", status="abstain"
        )

        assert response.score is None
        assert response.reasoning == "Cannot evaluate this criterion"
        assert response.status == "abstain"
        assert response.is_abstention()

    def test_score_clamping(self):
        """Test score is clamped to 1-10 range"""
        # Test upper bound
        response = EvaluationResponse(score=15.0)
        assert response.score == 10.0

        # Test lower bound
        response = EvaluationResponse(score=0.5)
        assert response.score == 1.0

        # Test valid range
        response = EvaluationResponse(score=7.5)
        assert response.score == 7.5

    def test_string_abstention_patterns(self):
        """Test string abstention patterns are handled"""
        patterns = ["NO_RESPONSE", "N/A", "ABSTAIN"]

        for pattern in patterns:
            response = EvaluationResponse(score=pattern)
            assert response.score is None
            assert response.status == "abstain"
            assert response.is_abstention()

    def test_legacy_format_conversion(self):
        """Test conversion to legacy (score, justification) format"""
        response = EvaluationResponse(
            score=8.0, reasoning="Excellent performance", status="success"
        )

        score, justification = response.get_legacy_format()
        assert score == 8.0
        assert justification == "Excellent performance"

    def test_from_dict_creation(self):
        """Test creation from dictionary"""
        data = {"score": 7.5, "reasoning": "Good option", "confidence": 0.8, "status": "success"}

        response = EvaluationResponse.from_dict(data)
        assert response.score == 7.5
        assert response.reasoning == "Good option"
        assert response.confidence == 0.8
        assert response.status == "success"

    def test_from_json_creation(self):
        """Test creation from JSON string"""
        json_str = '{"score": 6.0, "reasoning": "Average performance", "status": "success"}'

        response = EvaluationResponse.from_json(json_str)
        assert response.score == 6.0
        assert response.reasoning == "Average performance"
        assert response.status == "success"

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON"""
        with pytest.raises(ValueError, match="Invalid JSON format"):
            EvaluationResponse.from_json("invalid json")


class TestStructuredResponseParsing:
    """Test the main structured response parsing functionality"""

    def test_json_response_parsing(self):
        """Test parsing valid JSON responses"""
        json_response = """
        {
            "score": 8.5,
            "reasoning": "Excellent performance with minimal overhead",
            "confidence": 0.9,
            "status": "success"
        }
        """

        response = parse_structured_response(json_response)
        assert response.score == 8.5
        assert "Excellent performance" in response.reasoning
        assert response.confidence == 0.9
        assert response.status == "success"

    def test_json_with_extra_text(self):
        """Test JSON parsing when there's extra text around the JSON"""
        response_text = """
        Here's my evaluation:
        
        {
            "score": 7.0,
            "reasoning": "Good option but has some limitations",
            "status": "success"
        }
        
        That's my assessment.
        """

        response = parse_structured_response(response_text)
        assert response.score == 7.0
        assert "Good option but has some limitations" in response.reasoning

    def test_abstention_json_response(self):
        """Test JSON abstention responses"""
        json_response = """
        {
            "score": null,
            "reasoning": "This criterion is not applicable to this option",
            "status": "abstain"
        }
        """

        response = parse_structured_response(json_response)
        assert response.score is None
        assert response.status == "abstain"
        assert response.is_abstention()

    def test_fallback_to_legacy_parsing(self):
        """Test fallback to legacy regex parsing"""
        legacy_response = """
        SCORE: 8.5
        JUSTIFICATION: This option provides excellent performance with minimal overhead.
        """

        response = parse_structured_response(legacy_response)
        assert response.score == 8.5
        assert "excellent performance" in response.reasoning.lower()

    def test_legacy_abstention_fallback(self):
        """Test legacy abstention patterns in fallback"""
        legacy_response = """
        SCORE: [NO_RESPONSE]
        JUSTIFICATION: This criterion is not applicable to this option.
        """

        response = parse_structured_response(legacy_response)
        assert response.score is None
        assert response.is_abstention()
        assert "not applicable" in response.reasoning.lower()


class TestJSONExtraction:
    """Test JSON extraction from mixed content"""

    def test_extract_simple_json(self):
        """Test extracting simple JSON object"""
        text = 'Here is the result: {"score": 7.5, "status": "success"} done'
        json_str = _extract_json_from_response(text)

        assert json_str == '{"score": 7.5, "status": "success"}'

    def test_extract_nested_json(self):
        """Test extracting nested JSON object"""
        text = """
        Result:
        {
            "score": 8.0,
            "reasoning": "Good performance",
            "metadata": {
                "confidence": 0.8,
                "flags": ["reliable", "tested"]
            },
            "status": "success"
        }
        End of result.
        """

        json_str = _extract_json_from_response(text)
        parsed = json.loads(json_str)

        assert parsed["score"] == 8.0
        assert parsed["metadata"]["confidence"] == 0.8
        assert "reliable" in parsed["metadata"]["flags"]

    def test_no_json_found(self):
        """Test when no JSON is found"""
        text = "This is just plain text with no JSON objects"
        json_str = _extract_json_from_response(text)

        assert json_str is None

    def test_malformed_json_boundaries(self):
        """Test handling malformed JSON boundaries"""
        text = "Here is incomplete JSON: {score: 7.5, status"
        json_str = _extract_json_from_response(text)

        assert json_str is None


class TestLegacyCompatibility:
    """Test compatibility with existing legacy parsing"""

    def test_legacy_score_patterns(self):
        """Test all legacy score patterns still work"""
        test_cases = [
            ("SCORE: 8.5", 8.5),
            ("Score: 7/10", 7.0),
            ("Rating: 9.2", 9.2),
            ("Score = 6", 6.0),
            ("8.5/10", 8.5),
            ("The score is 7.5", 7.5),
            ("9", 9.0),  # Just a number
        ]

        for text, expected_score in test_cases:
            response = parse_structured_response(text)
            assert response.score == expected_score

    def test_legacy_justification_patterns(self):
        """Test legacy justification patterns"""
        test_cases = [
            ("SCORE: 7\nJUSTIFICATION: Good performance", "Good performance"),
            ("SCORE: 8\nReasoning: Works well", "Works well"),
            ("SCORE: 9\nExplanation: Handles edge cases", "Handles edge cases"),
            ("SCORE: 7\nBecause it's simple", "it's simple"),
        ]

        for text, expected_reasoning in test_cases:
            response = parse_structured_response(text)
            assert expected_reasoning in response.reasoning

    def test_legacy_abstention_patterns(self):
        """Test legacy abstention patterns still work"""
        patterns = [
            "[NO_RESPONSE]",
            "NO_RESPONSE - not applicable",
            "Cannot evaluate this criterion",
            "Not applicable to this option",
            "I must abstain from scoring",
        ]

        for pattern in patterns:
            response = parse_structured_response(pattern)
            assert response.score is None
            assert response.is_abstention()


class TestPromptGeneration:
    """Test JSON prompt generation"""

    def test_prompt_suffix_generation(self):
        """Test JSON prompt suffix contains required elements"""
        suffix = generate_json_prompt_suffix()

        # Check for required JSON structure
        assert "score" in suffix
        assert "reasoning" in suffix
        assert "confidence" in suffix
        assert "status" in suffix

        # Check for instructions
        assert "JSON object" in suffix
        assert "1-10" in suffix
        assert "null" in suffix
        assert "abstain" in suffix


class TestSchemaValidation:
    """Test response schema validation"""

    def test_valid_schema(self):
        """Test validation of valid schemas"""
        valid_data = {"score": 7.5, "reasoning": "Good performance", "status": "success"}

        assert validate_response_schema(valid_data) is True

    def test_invalid_schema_missing_fields(self):
        """Test validation fails for missing required fields"""
        invalid_data = {
            "score": 7.5,
            # Missing reasoning and status
        }

        assert validate_response_schema(invalid_data) is False

    def test_invalid_score_type(self):
        """Test validation fails for invalid score types"""
        invalid_data = {"score": "invalid", "reasoning": "Test", "status": "success"}

        assert validate_response_schema(invalid_data) is False

    def test_invalid_status_value(self):
        """Test validation fails for invalid status values"""
        invalid_data = {"score": 7.5, "reasoning": "Test", "status": "invalid_status"}

        assert validate_response_schema(invalid_data) is False


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_response(self):
        """Test handling of empty responses"""
        response = parse_structured_response("")
        assert response.score is None
        # Empty responses fall back to legacy parsing which returns a specific error message
        assert (
            "Could not parse evaluation from response" in response.reasoning
            or response.reasoning == "No justification provided"
        )

    def test_whitespace_only_response(self):
        """Test handling of whitespace-only responses"""
        response = parse_structured_response("   \n  \t  ")
        assert response.score is None

    def test_malformed_json_fallback(self):
        """Test malformed JSON falls back to regex parsing"""
        malformed = '{"score": 7.5, "reasoning": "test"'  # Missing closing brace
        response = parse_structured_response(malformed)

        # Should fall back and extract what it can
        assert isinstance(response, EvaluationResponse)

    def test_unicode_handling(self):
        """Test proper unicode handling"""
        json_response = """
        {
            "score": 8.5,
            "reasoning": "Excellent 💯 performance with ✨ features",
            "status": "success"
        }
        """

        response = parse_structured_response(json_response)
        assert response.score == 8.5
        assert "💯" in response.reasoning
        assert "✨" in response.reasoning

    def test_very_long_response(self):
        """Test handling of very long responses"""
        long_reasoning = "x" * 2000
        json_response = f"""
        {{
            "score": 7.0,
            "reasoning": "{long_reasoning}",
            "status": "success"
        }}
        """

        response = parse_structured_response(json_response)
        assert response.score == 7.0
        assert len(response.reasoning) == 2000  # Should preserve full reasoning in JSON mode
