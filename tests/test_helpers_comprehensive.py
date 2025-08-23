"""
Comprehensive tests for decision_matrix_mcp.helpers module.

This test suite provides 100% coverage for all helper functions including:
- Input validation and edge cases
- Error handling and defensive programming
- Response formatting consistency
- Object creation and manipulation
- Integration with models and data structures

Test Categories:
- Validation functions (prerequisites, inputs)
- Result processing (scores, abstentions, errors)
- Response creation (standardized formats)
- Object factory functions (criteria, sessions)
- Error scenarios and boundary conditions
"""

from unittest.mock import Mock, patch

# Import the functions we're testing
from decision_matrix_mcp.helpers import (
    create_criterion_from_request,
    create_criterion_response,
    create_error_response,
    create_evaluation_response,
    create_session_response,
    process_evaluation_results,
    process_initial_criteria,
    validate_evaluation_prerequisites,
)
from decision_matrix_mcp.models import DecisionSession, ModelBackend


class TestValidateEvaluationPrerequisites:
    """Test validation of evaluation prerequisites."""

    def test_valid_session_with_options_and_criteria(self):
        """Test that a valid session passes validation."""
        session = Mock(spec=DecisionSession)
        session.options = {"option1": Mock(), "option2": Mock()}
        session.criteria = {"criterion1": Mock(), "criterion2": Mock()}

        result = validate_evaluation_prerequisites(session)
        assert result is None

    def test_session_with_no_options(self):
        """Test error when session has no options."""
        session = Mock(spec=DecisionSession)
        session.options = {}
        session.criteria = {"criterion1": Mock()}

        result = validate_evaluation_prerequisites(session)
        assert result is not None
        assert result["error"] == "No options to evaluate. Add options first."
        assert result["validation_context"] == "prerequisites"

    def test_session_with_no_criteria(self):
        """Test error when session has no criteria."""
        session = Mock(spec=DecisionSession)
        session.options = {"option1": Mock()}
        session.criteria = {}

        result = validate_evaluation_prerequisites(session)
        assert result is not None
        assert result["error"] == "No criteria defined. Add criteria first."
        assert result["validation_context"] == "prerequisites"

    def test_session_with_empty_options_dict(self):
        """Test error when options dict is explicitly empty."""
        session = Mock(spec=DecisionSession)
        session.options = {}
        session.criteria = {"criterion1": Mock()}

        result = validate_evaluation_prerequisites(session)
        assert result["error"] == "No options to evaluate. Add options first."

    def test_session_with_empty_criteria_dict(self):
        """Test error when criteria dict is explicitly empty."""
        session = Mock(spec=DecisionSession)
        session.options = {"option1": Mock()}
        session.criteria = {}

        result = validate_evaluation_prerequisites(session)
        assert result["error"] == "No criteria defined. Add criteria first."

    def test_session_with_none_options(self):
        """Test error when options is None."""
        session = Mock(spec=DecisionSession)
        session.options = None
        session.criteria = {"criterion1": Mock()}

        result = validate_evaluation_prerequisites(session)
        assert result is not None
        assert "No options to evaluate" in result["error"]

    def test_session_with_none_criteria(self):
        """Test error when criteria is None."""
        session = Mock(spec=DecisionSession)
        session.options = {"option1": Mock()}
        session.criteria = None

        result = validate_evaluation_prerequisites(session)
        assert result is not None
        assert "No criteria defined" in result["error"]


class TestProcessEvaluationResults:
    """Test processing of evaluation results."""

    def test_successful_evaluation_results(self):
        """Test processing of normal evaluation results."""
        # Create mock session with options
        session = Mock(spec=DecisionSession)
        option1 = Mock()
        option2 = Mock()
        session.options = {"option1": option1, "option2": option2}

        evaluation_results = {
            "criterion1": {
                "option1": (8.5, "Good performance"),
                "option2": (7.0, "Decent performance"),
            },
            "criterion2": {
                "option1": (9.0, "Excellent features"),
                "option2": (6.5, "Basic features"),
            },
        }

        total_scores, abstentions, errors = process_evaluation_results(evaluation_results, session)

        assert total_scores == 4  # 2 criteria × 2 options
        assert abstentions == 0
        assert errors == []

        # Verify that scores were added to options
        assert option1.add_score.call_count == 2
        assert option2.add_score.call_count == 2

    def test_evaluation_results_with_abstentions(self):
        """Test processing results that include abstentions (None scores)."""
        session = Mock(spec=DecisionSession)
        option1 = Mock()
        option2 = Mock()
        session.options = {"option1": option1, "option2": option2}

        evaluation_results = {
            "criterion1": {
                "option1": (8.5, "Good performance"),
                "option2": (None, "[NO_RESPONSE]"),  # Abstention
            },
            "criterion2": {
                "option1": (None, "Not applicable"),  # Abstention
                "option2": (6.5, "Basic features"),
            },
        }

        total_scores, abstentions, errors = process_evaluation_results(evaluation_results, session)

        assert total_scores == 2  # Only non-None scores
        assert abstentions == 2  # Two abstentions
        assert errors == []

    def test_evaluation_results_with_errors(self):
        """Test processing results that include error messages."""
        session = Mock(spec=DecisionSession)
        option1 = Mock()
        option2 = Mock()
        session.options = {"option1": option1, "option2": option2}

        evaluation_results = {
            "criterion1": {
                "option1": (8.5, "Good performance"),
                "option2": (None, "Error: Connection timeout"),  # Error case
            },
            "criterion2": {
                "option1": (7.0, "Error: Invalid response format"),  # Error case with score
                "option2": (6.5, "Basic features"),
            },
        }

        total_scores, abstentions, errors = process_evaluation_results(evaluation_results, session)

        assert total_scores == 2  # Two scores without errors (8.5 and 6.5)
        assert abstentions == 0  # No abstentions (errors take precedence)
        assert len(errors) == 2  # Two error messages
        assert "criterion1→option2: Error: Connection timeout" in errors
        assert "criterion2→option1: Error: Invalid response format" in errors

    def test_evaluation_results_with_unknown_options(self):
        """Test processing results for options not in session."""
        session = Mock(spec=DecisionSession)
        option1 = Mock()
        session.options = {"option1": option1}  # Only has option1

        evaluation_results = {
            "criterion1": {
                "option1": (8.5, "Good performance"),
                "unknown_option": (7.0, "Should be ignored"),  # Unknown option
            }
        }

        total_scores, abstentions, errors = process_evaluation_results(evaluation_results, session)

        assert total_scores == 1  # Only option1 counted
        assert abstentions == 0
        assert errors == []
        # Only option1 should receive score
        assert option1.add_score.call_count == 1

    def test_evaluation_results_empty(self):
        """Test processing empty evaluation results."""
        session = Mock(spec=DecisionSession)
        session.options = {"option1": Mock()}

        evaluation_results = {}

        total_scores, abstentions, errors = process_evaluation_results(evaluation_results, session)

        assert total_scores == 0
        assert abstentions == 0
        assert errors == []

    def test_score_object_creation(self):
        """Test that Score objects are created correctly."""
        session = Mock(spec=DecisionSession)
        option1 = Mock()
        session.options = {"option1": option1}

        evaluation_results = {
            "criterion1": {
                "option1": (8.5, "Test justification"),
            }
        }

        with patch("decision_matrix_mcp.helpers.Score") as mock_score:
            mock_score_instance = Mock()
            mock_score.return_value = mock_score_instance

            process_evaluation_results(evaluation_results, session)

            # Verify Score was created with correct parameters
            mock_score.assert_called_once_with(
                criterion_name="criterion1",
                option_name="option1",
                score=8.5,
                justification="Test justification",
            )

            # Verify score was added to option
            option1.add_score.assert_called_once_with(mock_score_instance)


class TestCreateEvaluationResponse:
    """Test creation of evaluation responses."""

    def test_basic_evaluation_response(self):
        """Test creating a basic evaluation response."""
        session = Mock(spec=DecisionSession)
        session.options = {"opt1": Mock(), "opt2": Mock(), "opt3": Mock()}  # 3 options
        session.criteria = {"crit1": Mock(), "crit2": Mock()}  # 2 criteria

        response = create_evaluation_response(
            session_id="test-session-123", session=session, total_scores=6, abstentions=0, errors=[]
        )

        assert response["session_id"] == "test-session-123"
        assert response["evaluation_complete"] is True
        assert response["summary"]["options_evaluated"] == 3
        assert response["summary"]["criteria_used"] == 2
        assert response["summary"]["total_evaluations"] == 6  # 3 × 2
        assert response["summary"]["successful_scores"] == 6
        assert response["summary"]["abstentions"] == 0
        assert response["summary"]["errors"] == 0
        assert response["errors"] is None
        assert "3 options across 2 criteria" in response["message"]
        assert "get_decision_matrix" in response["next_steps"][0]

    def test_evaluation_response_with_errors_and_abstentions(self):
        """Test evaluation response with errors and abstentions."""
        session = Mock(spec=DecisionSession)
        session.options = {"opt1": Mock(), "opt2": Mock()}
        session.criteria = {"crit1": Mock()}

        errors = ["Error 1", "Error 2"]

        response = create_evaluation_response(
            session_id="test-session-456",
            session=session,
            total_scores=1,
            abstentions=1,
            errors=errors,
        )

        assert response["summary"]["successful_scores"] == 1
        assert response["summary"]["abstentions"] == 1
        assert response["summary"]["errors"] == 2
        assert response["errors"] == errors

    def test_evaluation_response_no_errors(self):
        """Test that errors field is None when no errors."""
        session = Mock(spec=DecisionSession)
        session.options = {"opt1": Mock()}
        session.criteria = {"crit1": Mock()}

        response = create_evaluation_response(
            session_id="test", session=session, total_scores=1, abstentions=0, errors=[]
        )

        assert response["errors"] is None

    def test_evaluation_response_empty_session(self):
        """Test response creation with empty session."""
        session = Mock(spec=DecisionSession)
        session.options = {}
        session.criteria = {}

        response = create_evaluation_response(
            session_id="empty-session", session=session, total_scores=0, abstentions=0, errors=[]
        )

        assert response["summary"]["options_evaluated"] == 0
        assert response["summary"]["criteria_used"] == 0
        assert response["summary"]["total_evaluations"] == 0
        assert "0 options across 0 criteria" in response["message"]


class TestCreateErrorResponse:
    """Test creation of error responses."""

    def test_basic_error_response(self):
        """Test creating basic error response without formatter."""
        response = create_error_response("Test error message")

        assert response["error"] == "Test error message"
        assert "formatted_output" not in response

    def test_error_response_with_context(self):
        """Test error response with context parameter."""
        response = create_error_response("Test error", context="Custom context")

        assert response["error"] == "Test error"
        # Context is used for formatter but not directly in response

    def test_error_response_with_formatter(self):
        """Test error response with formatter."""
        mock_formatter = Mock()
        mock_formatter.format_error.return_value = "Formatted error output"

        response = create_error_response(
            "Test error", context="Test context", formatter=mock_formatter
        )

        assert response["error"] == "Test error"
        assert response["formatted_output"] == "Formatted error output"
        mock_formatter.format_error.assert_called_once_with("Test error", "Test context")

    def test_error_response_with_none_formatter(self):
        """Test error response when formatter is explicitly None."""
        response = create_error_response("Test error", formatter=None)

        assert response["error"] == "Test error"
        assert "formatted_output" not in response

    def test_error_response_empty_message(self):
        """Test error response with empty error message."""
        response = create_error_response("")

        assert response["error"] == ""

    def test_error_response_default_context(self):
        """Test error response uses default context."""
        mock_formatter = Mock()
        mock_formatter.format_error.return_value = "Formatted"

        create_error_response("Test", formatter=mock_formatter)

        mock_formatter.format_error.assert_called_once_with("Test", "Validation error")


class TestProcessInitialCriteria:
    """Test processing of initial criteria from requests."""

    def test_process_initial_criteria_with_valid_data(self):
        """Test processing valid initial criteria."""
        request = Mock()
        request.initial_criteria = [
            {
                "name": "Performance",
                "description": "Speed and efficiency",
                "weight": 2.0,
                "temperature": 0.7,
            },
            {"name": "Cost", "description": "Financial impact", "weight": 1.5},
        ]
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = "claude-3"
        request.temperature = 0.5

        session = Mock(spec=DecisionSession)

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            mock_criterion_instance = Mock()
            mock_criterion.return_value = mock_criterion_instance

            criteria_added = process_initial_criteria(request, session)

            assert criteria_added == ["Performance", "Cost"]
            assert mock_criterion.call_count == 2
            assert session.add_criterion.call_count == 2

            # Check first criterion call
            first_call = mock_criterion.call_args_list[0]
            assert first_call[1]["name"] == "Performance"
            assert first_call[1]["description"] == "Speed and efficiency"
            assert first_call[1]["weight"] == 2.0
            assert first_call[1]["temperature"] == 0.7
            assert first_call[1]["model_backend"] == ModelBackend.BEDROCK
            assert first_call[1]["model_name"] == "claude-3"

            # Check second criterion call (uses default temperature)
            second_call = mock_criterion.call_args_list[1]
            assert second_call[1]["name"] == "Cost"
            assert second_call[1]["temperature"] == 0.5  # Default from request

    def test_process_initial_criteria_no_initial_criteria(self):
        """Test processing when no initial criteria provided."""
        request = Mock()
        request.initial_criteria = None

        session = Mock(spec=DecisionSession)

        criteria_added = process_initial_criteria(request, session)

        assert criteria_added == []
        assert not session.add_criterion.called

    def test_process_initial_criteria_empty_list(self):
        """Test processing when initial criteria is empty list."""
        request = Mock()
        request.initial_criteria = []

        session = Mock(spec=DecisionSession)

        criteria_added = process_initial_criteria(request, session)

        assert criteria_added == []
        assert not session.add_criterion.called

    def test_process_initial_criteria_partial_data(self):
        """Test processing criteria with partial/missing data."""
        request = Mock()
        request.initial_criteria = [
            {
                "name": "Quality",
                # Missing description, weight, temperature
            }
        ]
        request.model_backend = ModelBackend.LITELLM
        request.model_name = "gpt-4"
        request.temperature = 0.3

        session = Mock(spec=DecisionSession)

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            criteria_added = process_initial_criteria(request, session)

            assert criteria_added == ["Quality"]

            # Check defaults are applied
            call_args = mock_criterion.call_args[1]
            assert call_args["name"] == "Quality"
            assert call_args["description"] == ""  # Default empty string
            assert call_args["weight"] == 1.0  # Default weight
            assert call_args["temperature"] == 0.3  # From request

    def test_process_initial_criteria_with_empty_values(self):
        """Test processing criteria with empty string values."""
        request = Mock()
        request.initial_criteria = [
            {
                "name": "",  # Empty name
                "description": "",  # Empty description
                "weight": 0.0,  # Zero weight
            }
        ]
        request.model_backend = ModelBackend.OLLAMA
        request.model_name = "llama2"
        request.temperature = 0.8

        session = Mock(spec=DecisionSession)

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            criteria_added = process_initial_criteria(request, session)

            assert criteria_added == [""]  # Empty name is still processed

            call_args = mock_criterion.call_args[1]
            assert call_args["name"] == ""
            assert call_args["description"] == ""
            assert call_args["weight"] == 0.0


class TestCreateSessionResponse:
    """Test creation of session responses."""

    def test_basic_session_response(self):
        """Test creating basic session response."""
        session = Mock(spec=DecisionSession)
        session.session_id = "session-123"

        request = Mock()
        request.topic = "Test Decision"
        request.options = ["Option A", "Option B", "Option C"]
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = "claude-3-sonnet"

        criteria_added = ["Criterion 1", "Criterion 2"]

        response = create_session_response(session, request, criteria_added)

        assert response["session_id"] == "session-123"
        assert response["topic"] == "Test Decision"
        assert response["options"] == ["Option A", "Option B", "Option C"]
        assert response["criteria_added"] == ["Criterion 1", "Criterion 2"]
        assert response["model_backend"] == "bedrock"  # Enum value
        assert response["model_name"] == "claude-3-sonnet"
        assert "3 options and 2 criteria" in response["message"]
        assert any("add_criterion" in step for step in response["next_steps"])
        assert any("evaluate_options" in step for step in response["next_steps"])
        assert any("get_decision_matrix" in step for step in response["next_steps"])

    def test_session_response_no_criteria_added(self):
        """Test session response when no criteria were added."""
        session = Mock(spec=DecisionSession)
        session.session_id = "session-456"

        request = Mock()
        request.topic = "Simple Decision"
        request.options = ["Option 1"]
        request.model_backend = ModelBackend.LITELLM
        request.model_name = "gpt-4"

        criteria_added = []  # No criteria

        response = create_session_response(session, request, criteria_added)

        assert response["criteria_added"] == []
        assert "1 options" in response["message"]  # No criteria mentioned
        assert "and" not in response["message"]  # Should not mention criteria

    def test_session_response_enum_conversion(self):
        """Test that model backend enum is converted to string."""
        session = Mock(spec=DecisionSession)
        session.session_id = "session-789"

        request = Mock()
        request.topic = "Test"
        request.options = ["A"]
        request.model_backend = ModelBackend.OLLAMA  # Test different enum value
        request.model_name = "llama2"

        response = create_session_response(session, request, [])

        assert response["model_backend"] == "ollama"
        assert isinstance(response["model_backend"], str)

    def test_session_response_multiple_options_single_criterion(self):
        """Test message formatting with multiple options and single criterion."""
        session = Mock(spec=DecisionSession)
        session.session_id = "session-test"

        request = Mock()
        request.topic = "Multi-option Test"
        request.options = ["A", "B", "C", "D", "E"]
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = "titan"

        criteria_added = ["Single Criterion"]

        response = create_session_response(session, request, criteria_added)

        assert "5 options and 1 criteria" in response["message"]


class TestCreateCriterionFromRequest:
    """Test creation of Criterion objects from requests."""

    def test_basic_criterion_creation(self):
        """Test basic criterion creation from request."""
        request = Mock()
        request.name = "Performance"
        request.description = "Speed and efficiency metrics"
        request.weight = 2.5
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = "claude-3"
        request.temperature = 0.7
        request.custom_prompt = None

        session = Mock(spec=DecisionSession)
        session.default_temperature = 0.5

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            mock_criterion_instance = Mock()
            mock_criterion.return_value = mock_criterion_instance

            result = create_criterion_from_request(request, session)

            assert result == mock_criterion_instance
            mock_criterion.assert_called_once_with(
                name="Performance",
                description="Speed and efficiency metrics",
                weight=2.5,
                model_backend=ModelBackend.BEDROCK,
                model_name="claude-3",
                temperature=0.7,
            )

    def test_criterion_creation_with_none_temperature(self):
        """Test criterion creation when request temperature is None."""
        request = Mock()
        request.name = "Cost"
        request.description = "Financial considerations"
        request.weight = 1.0
        request.model_backend = ModelBackend.LITELLM
        request.model_name = "gpt-4"
        request.temperature = None  # Should use session default
        request.custom_prompt = None

        session = Mock(spec=DecisionSession)
        session.default_temperature = 0.3

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            create_criterion_from_request(request, session)

            call_args = mock_criterion.call_args[1]
            assert call_args["temperature"] == 0.3  # Session default

    def test_criterion_creation_with_custom_prompt(self):
        """Test criterion creation with custom system prompt."""
        request = Mock()
        request.name = "Quality"
        request.description = "Overall quality assessment"
        request.weight = 1.8
        request.model_backend = ModelBackend.OLLAMA
        request.model_name = "mistral"
        request.temperature = 0.6
        request.custom_prompt = "You are a quality assessment expert..."

        session = Mock(spec=DecisionSession)
        session.default_temperature = 0.5

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            mock_criterion_instance = Mock()
            mock_criterion.return_value = mock_criterion_instance

            create_criterion_from_request(request, session)

            # Verify custom prompt is set after creation
            assert mock_criterion_instance.system_prompt == "You are a quality assessment expert..."

    def test_criterion_creation_without_custom_prompt(self):
        """Test criterion creation without custom prompt."""
        request = Mock()
        request.name = "Usability"
        request.description = "User experience factors"
        request.weight = 1.2
        request.model_backend = ModelBackend.BEDROCK
        request.model_name = "claude-3-haiku"
        request.temperature = 0.4
        request.custom_prompt = None

        session = Mock(spec=DecisionSession)
        session.default_temperature = 0.5

        with patch("decision_matrix_mcp.models.Criterion") as mock_criterion:
            mock_criterion_instance = Mock()
            mock_criterion.return_value = mock_criterion_instance

            create_criterion_from_request(request, session)

            # Verify system_prompt is not modified
            assert (
                not hasattr(mock_criterion_instance, "system_prompt")
                or mock_criterion_instance.system_prompt != request.custom_prompt
            )


class TestCreateCriterionResponse:
    """Test creation of criterion addition responses."""

    def test_basic_criterion_response(self):
        """Test creating basic criterion response."""
        request = Mock()
        request.session_id = "session-abc"
        request.name = "Performance"
        request.description = "Speed and efficiency"
        request.weight = 2.0

        session = Mock(spec=DecisionSession)
        session.criteria = {"Performance": Mock(), "Cost": Mock(), "Quality": Mock()}

        response = create_criterion_response(request, session)

        assert response["session_id"] == "session-abc"
        assert response["criterion_added"] == "Performance"
        assert response["description"] == "Speed and efficiency"
        assert response["weight"] == 2.0
        assert response["total_criteria"] == 3
        assert set(response["all_criteria"]) == {"Performance", "Cost", "Quality"}
        assert "Performance" in response["message"]
        assert "2.0x" in response["message"]

    def test_criterion_response_first_criterion(self):
        """Test response when adding first criterion."""
        request = Mock()
        request.session_id = "session-first"
        request.name = "Initial Criterion"
        request.description = "First criterion added"
        request.weight = 1.0

        session = Mock(spec=DecisionSession)
        session.criteria = {"Initial Criterion": Mock()}  # Only one criterion

        response = create_criterion_response(request, session)

        assert response["total_criteria"] == 1
        assert response["all_criteria"] == ["Initial Criterion"]

    def test_criterion_response_weight_formatting(self):
        """Test different weight value formatting."""
        request = Mock()
        request.session_id = "session-weight"
        request.name = "Test Criterion"
        request.description = "Test"
        request.weight = 1.5

        session = Mock(spec=DecisionSession)
        session.criteria = {"Test Criterion": Mock()}

        response = create_criterion_response(request, session)

        assert "1.5x" in response["message"]

    def test_criterion_response_integer_weight(self):
        """Test response with integer weight."""
        request = Mock()
        request.session_id = "session-int"
        request.name = "Integer Weight"
        request.description = "Test integer"
        request.weight = 3

        session = Mock(spec=DecisionSession)
        session.criteria = {"Integer Weight": Mock()}

        response = create_criterion_response(request, session)

        assert "3x" in response["message"] or "3.0x" in response["message"]


# Integration tests to verify functions work together
class TestHelpersIntegration:
    """Integration tests for helper functions working together."""

    def test_complete_evaluation_workflow(self):
        """Test a complete workflow using multiple helper functions."""
        # Create a mock session with real structure
        session = Mock(spec=DecisionSession)
        option1 = Mock()
        option2 = Mock()
        session.options = {"Option A": option1, "Option B": option2}
        session.criteria = {"Criterion 1": Mock(), "Criterion 2": Mock()}

        # 1. Validate prerequisites
        validation_result = validate_evaluation_prerequisites(session)
        assert validation_result is None

        # 2. Process evaluation results
        evaluation_results = {
            "Criterion 1": {
                "Option A": (8.0, "Good choice"),
                "Option B": (7.0, "Decent option"),
            },
            "Criterion 2": {
                "Option A": (9.0, "Excellent"),
                "Option B": (None, "[NO_RESPONSE]"),  # Abstention
            },
        }

        total_scores, abstentions, errors = process_evaluation_results(evaluation_results, session)

        # 3. Create evaluation response
        response = create_evaluation_response(
            session_id="integration-test",
            session=session,
            total_scores=total_scores,
            abstentions=abstentions,
            errors=errors,
        )

        # Verify integration
        assert total_scores == 3
        assert abstentions == 1
        assert errors == []
        assert response["summary"]["successful_scores"] == 3
        assert response["summary"]["abstentions"] == 1

    def test_error_handling_integration(self):
        """Test error handling across helper functions."""
        # Create invalid session
        session = Mock(spec=DecisionSession)
        session.options = {}  # No options
        session.criteria = {"Test": Mock()}

        # Validate should fail
        validation_result = validate_evaluation_prerequisites(session)
        assert validation_result is not None

        # Create error response using the validation error
        error_response = create_error_response(validation_result["error"])

        assert "No options to evaluate" in error_response["error"]
