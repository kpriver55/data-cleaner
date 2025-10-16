"""
Unit tests for evaluation metrics

Run with: python -m pytest optimization/test_evaluators.py -v
"""

import pytest
import dspy
from unittest.mock import Mock
from optimization.evaluators import (
    OperationPresenceEvaluator,
    ParameterAccuracyEvaluator,
    ColumnSpecificityEvaluator,
    PlanStructureEvaluator,
    CompositeEvaluator,
    create_default_evaluator,
    dspy_metric,
    binary_metric
)


@pytest.fixture
def mock_io_tool():
    """Mock FileIOTool for testing"""
    return Mock()


@pytest.fixture
def sample_example():
    """Create a sample DSPy example for testing"""
    return dspy.Example(
        comprehensive_data_analysis="Sample data analysis",
        overall_cleaning_plan="""1. remove_duplicates(subset=['id', 'email'], keep='first')
2. handle_missing_values(columns=['age', 'salary'], numeric_strategy='median')
3. clean_text_columns(columns=['email'], operations=['strip', 'lower'])""",
        rationale="Remove duplicates, impute missing values, and normalize text"
    ).with_inputs('comprehensive_data_analysis')


class TestOperationPresenceEvaluator:
    """Tests for OperationPresenceEvaluator"""

    def test_perfect_match(self, mock_io_tool, sample_example):
        """Test perfect operation match"""
        evaluator = OperationPresenceEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = sample_example.overall_cleaning_plan

        score = evaluator.evaluate(sample_example, prediction)
        assert score == 1.0

    def test_missing_operation(self, mock_io_tool, sample_example):
        """Test with missing operation"""
        evaluator = OperationPresenceEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = """1. remove_duplicates(subset=['id'])
2. handle_missing_values(columns=['age'])"""

        score = evaluator.evaluate(sample_example, prediction)
        assert 0.0 < score < 1.0  # Should have partial score

    def test_extra_operation(self, mock_io_tool, sample_example):
        """Test with extra operation (lower precision)"""
        evaluator = OperationPresenceEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = """1. remove_duplicates(subset=['id'])
2. handle_missing_values(columns=['age'])
3. clean_text_columns(columns=['email'])
4. remove_outliers(columns=['salary'])"""

        score = evaluator.evaluate(sample_example, prediction)
        assert 0.0 < score < 1.0  # Perfect recall, but lower precision

    def test_no_operations(self, mock_io_tool):
        """Test with no operations in either plan"""
        evaluator = OperationPresenceEvaluator(mock_io_tool)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="No specific operations needed",
            rationale="Data is already clean"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "Nothing to do"

        score = evaluator.evaluate(example, prediction)
        assert score == 1.0


class TestParameterAccuracyEvaluator:
    """Tests for ParameterAccuracyEvaluator"""

    def test_correct_parameters(self, mock_io_tool, sample_example):
        """Test with correct parameters"""
        evaluator = ParameterAccuracyEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = sample_example.overall_cleaning_plan

        score = evaluator.evaluate(sample_example, prediction)
        assert score == 1.0

    def test_missing_parameters(self, mock_io_tool):
        """Test with missing parameters"""
        evaluator = ParameterAccuracyEvaluator(mock_io_tool)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. handle_missing_values(numeric_strategy='median', categorical_strategy='most_frequent')",
            rationale="Impute with median"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. handle_missing_values()"

        score = evaluator.evaluate(example, prediction)
        assert score < 1.0

    def test_no_parameters_required(self, mock_io_tool):
        """Test when no parameters are specified"""
        evaluator = ParameterAccuracyEvaluator(mock_io_tool)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. remove_duplicates()",
            rationale="Remove duplicates"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. remove_duplicates()"

        score = evaluator.evaluate(example, prediction)
        assert score == 1.0


class TestColumnSpecificityEvaluator:
    """Tests for ColumnSpecificityEvaluator"""

    def test_exact_column_match(self, mock_io_tool):
        """Test with exact column matches"""
        evaluator = ColumnSpecificityEvaluator(mock_io_tool)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. handle_missing_values(columns=['age', 'salary'])",
            rationale="Handle missing"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. handle_missing_values(columns=['age', 'salary'])"

        score = evaluator.evaluate(example, prediction)
        assert score == 1.0

    def test_missing_columns(self, mock_io_tool):
        """Test with missing column specifications"""
        evaluator = ColumnSpecificityEvaluator(mock_io_tool)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. handle_missing_values(columns=['age', 'salary'])",
            rationale="Handle missing"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. handle_missing_values(columns=['age'])"

        score = evaluator.evaluate(example, prediction)
        assert 0.0 < score < 1.0

    def test_no_columns_in_expected(self, mock_io_tool):
        """Test when expected plan doesn't specify columns"""
        evaluator = ColumnSpecificityEvaluator(mock_io_tool)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. handle_missing_values()",
            rationale="Handle missing"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. handle_missing_values(columns=['age'])"

        score = evaluator.evaluate(example, prediction)
        assert score >= 0.7  # Should still get decent score for having structure


class TestPlanStructureEvaluator:
    """Tests for PlanStructureEvaluator"""

    def test_well_structured_plan(self, mock_io_tool, sample_example):
        """Test with well-structured plan"""
        evaluator = PlanStructureEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = sample_example.overall_cleaning_plan

        score = evaluator.evaluate(sample_example, prediction)
        assert score == 1.0

    def test_poorly_structured_plan(self, mock_io_tool, sample_example):
        """Test with poorly structured plan"""
        evaluator = PlanStructureEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = "clean the data somehow"

        score = evaluator.evaluate(sample_example, prediction)
        assert score < 0.5

    def test_empty_plan(self, mock_io_tool, sample_example):
        """Test with empty plan"""
        evaluator = PlanStructureEvaluator(mock_io_tool)

        prediction = Mock()
        prediction.overall_cleaning_plan = ""

        score = evaluator.evaluate(sample_example, prediction)
        assert score < 0.3


class TestCompositeEvaluator:
    """Tests for CompositeEvaluator"""

    def test_composite_evaluation(self, mock_io_tool, sample_example):
        """Test composite evaluator with multiple sub-evaluators"""
        sub_evaluators = [
            (OperationPresenceEvaluator(mock_io_tool), 0.5),
            (ParameterAccuracyEvaluator(mock_io_tool), 0.5)
        ]
        evaluator = CompositeEvaluator(mock_io_tool, sub_evaluators)

        prediction = Mock()
        prediction.overall_cleaning_plan = sample_example.overall_cleaning_plan

        score = evaluator.evaluate(sample_example, prediction)
        assert 0.0 <= score <= 1.0

    def test_weight_normalization(self, mock_io_tool):
        """Test that weights are normalized"""
        sub_evaluators = [
            (OperationPresenceEvaluator(mock_io_tool), 2.0),
            (ParameterAccuracyEvaluator(mock_io_tool), 3.0)
        ]
        evaluator = CompositeEvaluator(mock_io_tool, sub_evaluators)

        # Check that weights sum to 1
        total_weight = sum(weight for _, weight in evaluator.evaluators)
        assert abs(total_weight - 1.0) < 1e-6


class TestMetricFunctions:
    """Tests for metric creation functions"""

    def test_dspy_metric_creation(self, mock_io_tool):
        """Test DSPy metric function creation"""
        metric = dspy_metric(mock_io_tool)

        assert callable(metric)

        # Test that it returns a float
        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. remove_duplicates()",
            rationale="Remove duplicates"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. remove_duplicates()"

        score = metric(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_binary_metric_creation(self, mock_io_tool):
        """Test binary metric function creation"""
        metric = binary_metric(mock_io_tool, threshold=0.7)

        assert callable(metric)

        # Test that it returns a bool
        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. remove_duplicates()",
            rationale="Remove duplicates"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. remove_duplicates()"

        result = metric(example, prediction)
        assert isinstance(result, bool)

    def test_binary_metric_threshold(self, mock_io_tool):
        """Test binary metric threshold behavior"""
        metric_low = binary_metric(mock_io_tool, threshold=0.1)
        metric_high = binary_metric(mock_io_tool, threshold=0.99)

        example = dspy.Example(
            comprehensive_data_analysis="Sample",
            overall_cleaning_plan="1. remove_duplicates()",
            rationale="Remove duplicates"
        ).with_inputs('comprehensive_data_analysis')

        prediction = Mock()
        prediction.overall_cleaning_plan = "1. remove_duplicates()"

        # Low threshold should pass
        assert metric_low(example, prediction) is True

        # High threshold might fail
        # (depends on evaluator, but should demonstrate threshold effect)


class TestDefaultEvaluator:
    """Tests for default evaluator creation"""

    def test_create_default_evaluator(self, mock_io_tool):
        """Test default evaluator creation"""
        evaluator = create_default_evaluator(mock_io_tool)

        assert isinstance(evaluator, CompositeEvaluator)
        assert len(evaluator.evaluators) > 0

        # Check that weights sum to 1
        total_weight = sum(weight for _, weight in evaluator.evaluators)
        assert abs(total_weight - 1.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
