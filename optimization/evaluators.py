"""
Evaluation Metrics for Data Cleaning Quality

Provides metrics to assess the quality of data cleaning operations
for use in DSPy optimization.

Focus: Plan quality and execution correctness (what actually affects output)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Set
from pathlib import Path
import dspy


class CleaningEvaluator:
    """
    Base class for evaluating data cleaning quality
    """

    def __init__(self, io_tool):
        """
        Initialize evaluator

        Args:
            io_tool: FileIOTool instance for reading datasets
        """
        self.io_tool = io_tool

    def evaluate(self, example, prediction, trace=None) -> float:
        """
        Evaluate a single prediction

        Args:
            example: DSPy Example object with inputs and expected outputs
            prediction: DSPy Prediction object with model outputs
            trace: Optional trace information from DSPy

        Returns:
            Score between 0 and 1 (1 is perfect)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class OperationPresenceEvaluator(CleaningEvaluator):
    """
    Evaluates whether all expected operations are mentioned in the plan

    This is a fundamental check - did the plan include all necessary operations?
    """

    def evaluate(self, example, prediction, trace=None) -> float:
        """
        Evaluate operation presence

        Args:
            example: DSPy Example with expected overall_cleaning_plan
            prediction: DSPy Prediction with predicted overall_cleaning_plan
            trace: Optional trace

        Returns:
            Score between 0 and 1 based on operation coverage
        """
        expected_plan = example.overall_cleaning_plan.lower()
        predicted_plan = prediction.overall_cleaning_plan.lower()

        # Extract operation names from expected plan
        expected_operations = self._extract_operations(expected_plan)
        predicted_operations = self._extract_operations(predicted_plan)

        if not expected_operations:
            return 1.0  # No specific operations required

        # Calculate recall: what fraction of expected operations are present?
        expected_set = set(expected_operations)
        predicted_set = set(predicted_operations)

        recall = len(expected_set & predicted_set) / len(expected_set)

        # Calculate precision: what fraction of predicted operations are correct?
        if predicted_set:
            precision = len(expected_set & predicted_set) / len(predicted_set)
        else:
            precision = 0.0

        # F1 score balances precision and recall
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def _extract_operations(self, plan: str) -> List[str]:
        """Extract operation names from a plan string"""
        operations = [
            'remove_duplicates',
            'handle_missing_values',
            'remove_outliers',
            'clean_text_columns',
            'convert_data_types',
        ]
        return [op for op in operations if op in plan]


class ParameterAccuracyEvaluator(CleaningEvaluator):
    """
    Evaluates whether the plan specifies correct parameters

    Checks for mentions of expected strategies, methods, and thresholds.
    """

    def evaluate(self, example, prediction, trace=None) -> float:
        """
        Evaluate parameter accuracy

        Args:
            example: DSPy Example with expected plan
            prediction: DSPy Prediction with predicted plan
            trace: Optional trace

        Returns:
            Score between 0 and 1 based on parameter matching
        """
        expected_plan = example.overall_cleaning_plan.lower()
        predicted_plan = prediction.overall_cleaning_plan.lower()

        # Extract parameter keywords
        expected_params = self._extract_parameters(expected_plan)
        predicted_params = self._extract_parameters(predicted_plan)

        if not expected_params:
            return 1.0  # No specific parameters required

        # Calculate how many expected parameters are present
        expected_set = set(expected_params)
        predicted_set = set(predicted_params)

        if not expected_set:
            return 1.0

        accuracy = len(expected_set & predicted_set) / len(expected_set)
        return accuracy

    def _extract_parameters(self, plan: str) -> List[str]:
        """Extract parameter keywords from a plan string"""
        # Imputation strategies
        strategies = ['median', 'mean', 'mode', 'most_frequent', 'forward_fill',
                     'backward_fill', 'knn', 'drop']

        # Outlier detection methods
        methods = ['iqr', 'zscore', 'isolation_forest', 'modified_zscore']

        # Text operations
        text_ops = ['strip', 'lower', 'upper', 'remove_special_chars']

        # Other parameters
        others = ['first', 'last', 'threshold']

        all_params = strategies + methods + text_ops + others
        return [param for param in all_params if param in plan]


class ColumnSpecificityEvaluator(CleaningEvaluator):
    """
    Evaluates whether the plan specifies which columns operations apply to

    This is crucial for quality - good plans specify exact columns, not just operations.
    """

    def evaluate(self, example, prediction, trace=None) -> float:
        """
        Evaluate column specificity

        Args:
            example: DSPy Example with expected plan
            prediction: DSPy Prediction with predicted plan
            trace: Optional trace

        Returns:
            Score between 0 and 1 based on column mentions
        """
        expected_plan = example.overall_cleaning_plan
        predicted_plan = prediction.overall_cleaning_plan

        # Extract column names mentioned in expected plan
        expected_columns = self._extract_column_mentions(expected_plan)
        predicted_columns = self._extract_column_mentions(predicted_plan)

        if not expected_columns:
            # If expected plan doesn't specify columns, check that predicted has some structure
            # Look for "columns=" or column list patterns
            has_column_structure = 'columns=' in predicted_plan or '[' in predicted_plan
            return 1.0 if has_column_structure else 0.7

        # Calculate overlap of mentioned columns
        expected_set = set(expected_columns)
        predicted_set = set(predicted_columns)

        if not expected_set:
            return 1.0

        # Calculate recall: how many expected columns are mentioned?
        recall = len(expected_set & predicted_set) / len(expected_set)

        return recall

    def _extract_column_mentions(self, plan: str) -> List[str]:
        """
        Extract column names from a plan

        This looks for patterns like:
        - columns=['age', 'salary']
        - columns=[age, salary]
        - subset=['id', 'email']
        """
        import re

        columns = []

        # Pattern 1: columns=['col1', 'col2'] or columns=["col1", "col2"]
        pattern1 = r"columns=\[([^\]]+)\]"
        matches1 = re.findall(pattern1, plan)
        for match in matches1:
            # Extract quoted strings
            cols = re.findall(r"['\"]([^'\"]+)['\"]", match)
            columns.extend(cols)

        # Pattern 2: subset=['col1', 'col2']
        pattern2 = r"subset=\[([^\]]+)\]"
        matches2 = re.findall(pattern2, plan)
        for match in matches2:
            cols = re.findall(r"['\"]([^'\"]+)['\"]", match)
            columns.extend(cols)

        return columns


class PlanStructureEvaluator(CleaningEvaluator):
    """
    Evaluates the overall structure and completeness of the plan

    Checks that the plan is well-formed with numbered steps and clear operations.
    """

    def evaluate(self, example, prediction, trace=None) -> float:
        """
        Evaluate plan structure

        Args:
            example: DSPy Example
            prediction: DSPy Prediction with predicted plan
            trace: Optional trace

        Returns:
            Score between 0 and 1 based on structure quality
        """
        predicted_plan = prediction.overall_cleaning_plan

        score = 0.0
        checks = 0

        # Check 1: Has reasonable length (not too short, not empty)
        if len(predicted_plan.strip()) > 20:
            score += 1.0
        elif len(predicted_plan.strip()) > 0:
            score += 0.3
        checks += 1

        # Check 2: Has numbered steps (indicates structured plan)
        import re
        has_numbering = bool(re.search(r'^\s*\d+\.', predicted_plan, re.MULTILINE))
        if has_numbering:
            score += 1.0
        checks += 1

        # Check 3: Contains function call syntax operation(...)
        has_function_calls = '(' in predicted_plan and ')' in predicted_plan
        if has_function_calls:
            score += 1.0
        checks += 1

        # Check 4: Contains parameter specifications (=)
        has_parameters = '=' in predicted_plan
        if has_parameters:
            score += 1.0
        checks += 1

        return score / checks if checks > 0 else 0.0


class CompositeEvaluator(CleaningEvaluator):
    """
    Combines multiple evaluators with weights

    This allows for comprehensive evaluation using multiple metrics.
    """

    def __init__(
        self,
        io_tool,
        evaluators: List[tuple[CleaningEvaluator, float]]
    ):
        """
        Initialize composite evaluator

        Args:
            io_tool: FileIOTool instance
            evaluators: List of (evaluator, weight) tuples
        """
        super().__init__(io_tool)
        self.evaluators = evaluators

        # Normalize weights
        total_weight = sum(weight for _, weight in evaluators)
        if total_weight > 0:
            self.evaluators = [
                (evaluator, weight / total_weight)
                for evaluator, weight in evaluators
            ]

    def evaluate(self, example, prediction, trace=None) -> float:
        """
        Evaluate using all sub-evaluators

        Args:
            example: DSPy Example
            prediction: DSPy Prediction
            trace: Optional trace

        Returns:
            Weighted average of all evaluator scores
        """
        total_score = 0.0

        for evaluator, weight in self.evaluators:
            score = evaluator.evaluate(example, prediction, trace)
            total_score += score * weight

        return total_score


def create_default_evaluator(io_tool) -> CleaningEvaluator:
    """
    Create a default composite evaluator with reasonable settings

    Focuses on what matters: operations, parameters, and column specificity

    Args:
        io_tool: FileIOTool instance

    Returns:
        CompositeEvaluator instance
    """
    evaluators = [
        (OperationPresenceEvaluator(io_tool), 0.35),      # Most important: right operations
        (ColumnSpecificityEvaluator(io_tool), 0.30),      # Very important: column-specific
        (ParameterAccuracyEvaluator(io_tool), 0.25),      # Important: right parameters
        (PlanStructureEvaluator(io_tool), 0.10),          # Less important: structure
    ]

    return CompositeEvaluator(io_tool, evaluators)


def dspy_metric(io_tool, evaluator: Optional[CleaningEvaluator] = None) -> Callable:
    """
    Create a DSPy-compatible metric function

    DSPy optimizers expect a metric function with signature:
        metric(example, prediction, trace=None) -> float or bool

    Args:
        io_tool: FileIOTool instance
        evaluator: Optional CleaningEvaluator to use (default: composite evaluator)

    Returns:
        Metric function compatible with DSPy optimizers
    """
    if evaluator is None:
        evaluator = create_default_evaluator(io_tool)

    def metric(example, prediction, trace=None) -> float:
        """
        DSPy metric function

        Args:
            example: DSPy Example with expected outputs
            prediction: DSPy Prediction with model outputs
            trace: Optional execution trace

        Returns:
            Score between 0 and 1
        """
        try:
            return evaluator.evaluate(example, prediction, trace)
        except Exception as e:
            print(f"Error in metric evaluation: {e}")
            return 0.0

    return metric


def binary_metric(io_tool, threshold: float = 0.7, evaluator: Optional[CleaningEvaluator] = None) -> Callable:
    """
    Create a binary (pass/fail) DSPy metric

    Some DSPy optimizers work better with binary metrics.

    Args:
        io_tool: FileIOTool instance
        threshold: Score threshold for passing (default: 0.7)
        evaluator: Optional CleaningEvaluator to use

    Returns:
        Binary metric function
    """
    if evaluator is None:
        evaluator = create_default_evaluator(io_tool)

    def metric(example, prediction, trace=None) -> bool:
        """
        Binary DSPy metric

        Args:
            example: DSPy Example with expected outputs
            prediction: DSPy Prediction with model outputs
            trace: Optional execution trace

        Returns:
            True if score >= threshold, False otherwise
        """
        try:
            score = evaluator.evaluate(example, prediction, trace)
            return score >= threshold
        except Exception as e:
            print(f"Error in metric evaluation: {e}")
            return False

    return metric


# TODO: Implement execution-based evaluation
# This would actually run the cleaning agent and compare outputs:
#
# class ExecutionCorrectnessEvaluator(CleaningEvaluator):
#     """
#     Evaluates actual cleaning results by running the agent
#
#     This would:
#     1. Take the predicted plan
#     2. Run the full cleaning pipeline
#     3. Compare output to expected cleaned dataset
#     4. Score based on data similarity metrics:
#        - Same number of rows
#        - Same columns
#        - Same data types
#        - Value-level similarity
#
#     Challenges:
#     - Expensive (requires full agent execution)
#     - Need to integrate plan execution into agent
#     - May have non-deterministic results
#     """
#     pass
