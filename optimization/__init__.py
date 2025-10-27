"""
DSPy Optimization Module for Data Cleaning Agent

This module provides tools for optimizing the data cleaning agent using DSPy's
optimization capabilities.

Main components:
- dataset: Training/test example management
- evaluators: Metrics for assessing cleaning quality
- config: Configuration management for optimization runs
- optimizers: DSPy optimizer wrappers
- benchmarking: Performance benchmarking tools
"""

from .benchmarking import Benchmark, BenchmarkMetrics, BenchmarkResult
from .config import (
    EvaluationConfig,
    OptimizationConfig,
    OptimizerConfig,
    create_balanced_optimization_config,
    create_quick_optimization_config,
    create_thorough_optimization_config,
)
from .dataset import CleaningDataset, CleaningExample
from .evaluators import (
    CleaningEvaluator,
    ColumnSpecificityEvaluator,
    CompositeEvaluator,
    OperationPresenceEvaluator,
    ParameterAccuracyEvaluator,
    PlanStructureEvaluator,
    binary_metric,
    create_default_evaluator,
    dspy_metric,
)
from .optimizers import (
    BootstrapFewShotOptimizer,
    BootstrapFewShotWithRandomSearchOptimizer,
    MIPROOptimizer,
    OptimizationResult,
    OptimizerWrapper,
    create_optimizer,
)

__all__ = [
    # Dataset management
    "CleaningExample",
    "CleaningDataset",
    # Evaluators
    "CleaningEvaluator",
    "OperationPresenceEvaluator",
    "ParameterAccuracyEvaluator",
    "ColumnSpecificityEvaluator",
    "PlanStructureEvaluator",
    "CompositeEvaluator",
    "create_default_evaluator",
    "dspy_metric",
    "binary_metric",
    # Configuration
    "OptimizerConfig",
    "EvaluationConfig",
    "OptimizationConfig",
    "create_quick_optimization_config",
    "create_thorough_optimization_config",
    "create_balanced_optimization_config",
    # Optimizers
    "OptimizerWrapper",
    "BootstrapFewShotOptimizer",
    "BootstrapFewShotWithRandomSearchOptimizer",
    "MIPROOptimizer",
    "create_optimizer",
    "OptimizationResult",
    # Benchmarking
    "Benchmark",
    "BenchmarkMetrics",
    "BenchmarkResult",
]

__version__ = "0.2.0"
