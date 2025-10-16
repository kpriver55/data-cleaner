"""
DSPy Optimizer Wrappers

Provides unified interface for different DSPy optimizers with progress tracking,
error handling, and configuration management.
"""

import dspy
from typing import List, Callable, Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import time
from datetime import datetime

from .config import OptimizerConfig


class OptimizerWrapper(ABC):
    """
    Base wrapper class for DSPy optimizers

    Provides unified interface, progress tracking, and error handling
    """

    def __init__(self, config: OptimizerConfig):
        """
        Initialize optimizer wrapper

        Args:
            config: OptimizerConfig with optimizer settings
        """
        self.config = config
        self.optimization_history = []
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def _create_optimizer(self, metric: Callable, **kwargs) -> Any:
        """
        Create the underlying DSPy optimizer instance

        Args:
            metric: Evaluation metric function
            **kwargs: Additional optimizer-specific arguments

        Returns:
            DSPy optimizer instance
        """
        pass

    def optimize(
        self,
        student_module: dspy.Module,
        trainset: List[dspy.Example],
        metric: Callable,
        valset: Optional[List[dspy.Example]] = None,
        teacher_module: Optional[dspy.Module] = None,
        verbose: bool = True
    ) -> dspy.Module:
        """
        Run optimization and return compiled module

        Args:
            student_module: The module to optimize (e.g., DataCleaningAgent.analyzer)
            trainset: Training examples
            metric: Evaluation metric function
            valset: Optional validation set
            teacher_module: Optional teacher module for generating training data
            verbose: Whether to print progress

        Returns:
            Optimized (compiled) module
        """
        if verbose:
            print(f"\n🚀 Starting {self.__class__.__name__} optimization")
            print(f"   Training examples: {len(trainset)}")
            if valset:
                print(f"   Validation examples: {len(valset)}")
            print(f"   Configuration: {self.config.to_dict()}")

        self.start_time = time.time()

        try:
            # Create optimizer
            optimizer = self._create_optimizer(metric, teacher=teacher_module)

            # Run compilation
            if verbose:
                print(f"\n⚙️  Compiling module...")

            compiled_module = optimizer.compile(
                student_module,
                trainset=trainset,
                valset=valset
            )

            self.end_time = time.time()
            duration = self.end_time - self.start_time

            if verbose:
                print(f"\n✅ Optimization completed in {duration:.2f} seconds")

            # Store optimization info
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'train_size': len(trainset),
                'val_size': len(valset) if valset else 0,
                'config': self.config.to_dict()
            })

            return compiled_module

        except Exception as e:
            self.end_time = time.time()
            if verbose:
                print(f"\n❌ Optimization failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics

        Returns:
            Dictionary with stats
        """
        if not self.start_time:
            return {'status': 'not_started'}

        stats = {
            'optimizer_type': self.__class__.__name__,
            'config': self.config.to_dict(),
            'history': self.optimization_history
        }

        if self.end_time:
            stats['last_duration_seconds'] = self.end_time - self.start_time
            stats['status'] = 'completed'
        else:
            stats['status'] = 'running'

        return stats


class BootstrapFewShotOptimizer(OptimizerWrapper):
    """
    Wrapper for dspy.BootstrapFewShot

    Simple and effective optimizer that creates few-shot demonstrations
    by running the teacher model on training examples.

    Good for: Quick optimization, small datasets, when you have labeled examples
    """

    def _create_optimizer(self, metric: Callable, **kwargs) -> dspy.BootstrapFewShot:
        """
        Create BootstrapFewShot optimizer

        Args:
            metric: Evaluation metric
            **kwargs: Additional arguments (e.g., teacher)

        Returns:
            dspy.BootstrapFewShot instance
        """
        teacher = kwargs.get('teacher')

        return dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
            teacher_settings=self.config.teacher_settings or {},
            max_rounds=1  # Standard setting
        )


class BootstrapFewShotWithRandomSearchOptimizer(OptimizerWrapper):
    """
    Wrapper for dspy.BootstrapFewShotWithRandomSearch

    Extends BootstrapFewShot with random search over candidate programs.
    More thorough than basic BootstrapFewShot.

    Good for: Better optimization, moderate-sized datasets
    """

    def _create_optimizer(
        self,
        metric: Callable,
        **kwargs
    ) -> dspy.BootstrapFewShotWithRandomSearch:
        """
        Create BootstrapFewShotWithRandomSearch optimizer

        Args:
            metric: Evaluation metric
            **kwargs: Additional arguments

        Returns:
            dspy.BootstrapFewShotWithRandomSearch instance
        """
        teacher = kwargs.get('teacher')

        return dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
            num_candidate_programs=self.config.num_candidate_programs,
            num_threads=self.config.num_threads,
            teacher_settings=self.config.teacher_settings or {}
        )


class MIPROOptimizer(OptimizerWrapper):
    """
    Wrapper for dspy.MIPRO (or MIPROv2)

    Multi-stage Instruction Proposal and Refinement Optimizer.
    Most sophisticated optimizer that optimizes both instructions and demonstrations.

    Good for: Production models, large datasets, best quality results
    Note: Slower than BootstrapFewShot but typically produces better results
    """

    def _create_optimizer(self, metric: Callable, **kwargs) -> Any:
        """
        Create MIPRO optimizer

        Args:
            metric: Evaluation metric
            **kwargs: Additional arguments

        Returns:
            dspy.MIPRO or dspy.MIPROv2 instance
        """
        # Try to use MIPROv2 if available, fall back to MIPRO
        try:
            mipro_class = dspy.MIPROv2
        except AttributeError:
            mipro_class = dspy.MIPRO

        return mipro_class(
            metric=metric,
            num_candidates=self.config.num_candidate_programs,
            init_temperature=0.7,  # Standard setting
            verbose=True
        )


def create_optimizer(config: OptimizerConfig) -> OptimizerWrapper:
    """
    Factory function to create optimizer based on config

    Args:
        config: OptimizerConfig specifying optimizer type and settings

    Returns:
        OptimizerWrapper instance

    Raises:
        ValueError: If optimizer_type is not recognized
    """
    optimizer_map = {
        'BootstrapFewShot': BootstrapFewShotOptimizer,
        'BootstrapFewShotWithRandomSearch': BootstrapFewShotWithRandomSearchOptimizer,
        'MIPRO': MIPROOptimizer,
        'MIPROv2': MIPROOptimizer  # Use same wrapper for v2
    }

    optimizer_class = optimizer_map.get(config.optimizer_type)

    if optimizer_class is None:
        raise ValueError(
            f"Unknown optimizer type: {config.optimizer_type}. "
            f"Available types: {list(optimizer_map.keys())}"
        )

    return optimizer_class(config)


class OptimizationResult:
    """
    Container for optimization results and metadata
    """

    def __init__(
        self,
        compiled_module: dspy.Module,
        config: OptimizerConfig,
        train_score: float,
        val_score: Optional[float] = None,
        train_size: int = 0,
        val_size: int = 0,
        duration_seconds: float = 0.0,
        optimizer_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize optimization result

        Args:
            compiled_module: The optimized DSPy module
            config: Configuration used for optimization
            train_score: Score on training set
            val_score: Score on validation set (if available)
            train_size: Number of training examples
            val_size: Number of validation examples
            duration_seconds: Time taken for optimization
            optimizer_stats: Additional statistics from optimizer
        """
        self.compiled_module = compiled_module
        self.config = config
        self.train_score = train_score
        self.val_score = val_score
        self.train_size = train_size
        self.val_size = val_size
        self.duration_seconds = duration_seconds
        self.optimizer_stats = optimizer_stats or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary (for serialization)

        Returns:
            Dictionary representation
        """
        return {
            'timestamp': self.timestamp,
            'config': self.config.to_dict(),
            'metrics': {
                'train_score': self.train_score,
                'val_score': self.val_score,
                'train_size': self.train_size,
                'val_size': self.val_size
            },
            'duration_seconds': self.duration_seconds,
            'optimizer_stats': self.optimizer_stats
        }

    def summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Summary string
        """
        lines = [
            "=" * 60,
            "Optimization Results",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Optimizer: {self.config.optimizer_type}",
            f"Duration: {self.duration_seconds:.2f} seconds",
            "",
            "Scores:",
            f"  Training   ({self.train_size} examples): {self.train_score:.4f}",
        ]

        if self.val_score is not None:
            lines.append(f"  Validation ({self.val_size} examples): {self.val_score:.4f}")

        lines.extend([
            "",
            "Configuration:",
            f"  Max bootstrapped demos: {self.config.max_bootstrapped_demos}",
            f"  Max labeled demos: {self.config.max_labeled_demos}",
            f"  Num candidate programs: {self.config.num_candidate_programs}",
            f"  Num threads: {self.config.num_threads}",
            "=" * 60
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation"""
        return f"OptimizationResult(train={self.train_score:.4f}, val={self.val_score:.4f if self.val_score else 'N/A'})"
