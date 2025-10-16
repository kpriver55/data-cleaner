"""
DSPy Optimization Module for Data Cleaning Agent

This module provides tools for optimizing the data cleaning agent using DSPy's
optimization capabilities.

Main components:
- dataset: Training/test example management
- evaluators: Metrics for assessing cleaning quality
- config: Configuration management for optimization runs
- optimizers: DSPy optimizer wrappers (to be implemented in Phase 2)
"""

from .dataset import CleaningExample, CleaningDataset
from .evaluators import (
    CleaningEvaluator,
    OperationPresenceEvaluator,
    ParameterAccuracyEvaluator,
    ColumnSpecificityEvaluator,
    PlanStructureEvaluator,
    CompositeEvaluator,
    create_default_evaluator,
    dspy_metric,
    binary_metric
)
from .config import (
    OptimizerConfig,
    EvaluationConfig,
    OptimizationConfig,
    create_quick_optimization_config,
    create_thorough_optimization_config,
    create_balanced_optimization_config
)

__all__ = [
    # Dataset management
    'CleaningExample',
    'CleaningDataset',

    # Evaluators
    'CleaningEvaluator',
    'OperationPresenceEvaluator',
    'ParameterAccuracyEvaluator',
    'ColumnSpecificityEvaluator',
    'PlanStructureEvaluator',
    'CompositeEvaluator',
    'create_default_evaluator',
    'dspy_metric',
    'binary_metric',

    # Configuration
    'OptimizerConfig',
    'EvaluationConfig',
    'OptimizationConfig',
    'create_quick_optimization_config',
    'create_thorough_optimization_config',
    'create_balanced_optimization_config',
]

__version__ = '0.1.0'
