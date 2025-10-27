"""
Benchmarking tools for DSPy optimization.

This module provides tools to benchmark different optimization strategies,
measure performance, and compare results across different configurations.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from optimization.config import OptimizationConfig
from optimization.dataset import CleaningDataset
from optimization.evaluators import CleaningEvaluator, create_default_evaluator


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run"""

    # Timing metrics
    total_time: float  # Total wall-clock time (seconds)
    optimization_time: float  # Time spent in optimizer
    evaluation_time: float  # Time spent in evaluation

    # Quality metrics
    train_score: float  # Average score on training set
    val_score: float  # Average score on validation set
    test_score: Optional[float] = None  # Average score on test set (if available)

    # Optimizer-specific metrics
    num_examples_used: int = 0  # Number of training examples used
    num_demonstrations: int = 0  # Number of demos in final model
    num_iterations: int = 0  # Number of optimizer iterations

    # Resource metrics
    memory_peak_mb: Optional[float] = None  # Peak memory usage
    num_llm_calls: Optional[int] = None  # Total LLM API calls

    # Additional metadata
    optimizer_name: str = ""
    config_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run"""

    config: Dict[str, Any]  # Configuration used
    metrics: BenchmarkMetrics
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)  # Per-example results
    errors: List[str] = field(default_factory=list)  # Any errors encountered

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "config": self.config,
            "metrics": {
                "total_time": self.metrics.total_time,
                "optimization_time": self.metrics.optimization_time,
                "evaluation_time": self.metrics.evaluation_time,
                "train_score": self.metrics.train_score,
                "val_score": self.metrics.val_score,
                "test_score": self.metrics.test_score,
                "num_examples_used": self.metrics.num_examples_used,
                "num_demonstrations": self.metrics.num_demonstrations,
                "num_iterations": self.metrics.num_iterations,
                "memory_peak_mb": self.metrics.memory_peak_mb,
                "num_llm_calls": self.metrics.num_llm_calls,
                "optimizer_name": self.metrics.optimizer_name,
                "config_name": self.metrics.config_name,
                "timestamp": self.metrics.timestamp,
            },
            "detailed_results": self.detailed_results,
            "errors": self.errors,
        }

    def save(self, output_path: Path):
        """Save benchmark result to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Benchmark:
    """
    Benchmarking harness for DSPy optimization.

    Measures performance across different optimizers, datasets, and configurations.
    """

    def __init__(self, output_dir: Path = Path("benchmarks")):
        """
        Initialize benchmark harness.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def run_single_benchmark(
        self,
        agent,
        config: OptimizationConfig,
        evaluator: Optional[CleaningEvaluator] = None,
        test_dataset: Optional[CleaningDataset] = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run a single benchmark with given configuration.

        Args:
            agent: DataCleaningAgent instance
            config: Optimization configuration to benchmark
            evaluator: Custom evaluator (uses default if None)
            test_dataset: Optional test set for final evaluation
            verbose: Print progress information

        Returns:
            BenchmarkResult with metrics and detailed results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {config.optimizer_name}")
            print(f"Config: {config.name or 'custom'}")
            print(f"{'='*60}\n")

        # Start timing
        start_time = time.time()

        # Create evaluator if not provided
        if evaluator is None:
            evaluator = create_default_evaluator(agent.io_tool)

        # Run optimization
        opt_start = time.time()
        try:
            result = agent.compile_with_optimizer(config, verbose=verbose)
            opt_time = time.time() - opt_start

            # Extract metrics from optimization result
            metrics = BenchmarkMetrics(
                total_time=0,  # Will be set later
                optimization_time=opt_time,
                evaluation_time=0,  # Will be set later
                train_score=result.train_score,
                val_score=result.val_score,
                num_examples_used=result.num_train_examples,
                num_demonstrations=result.num_bootstrapped_demos,
                optimizer_name=config.optimizer_name,
                config_name=config.name or "custom",
            )

            errors = []

        except Exception as e:
            opt_time = time.time() - opt_start
            if verbose:
                print(f"❌ Optimization failed: {e}")

            # Create metrics with zeros
            metrics = BenchmarkMetrics(
                total_time=0,
                optimization_time=opt_time,
                evaluation_time=0,
                train_score=0.0,
                val_score=0.0,
                optimizer_name=config.optimizer_name,
                config_name=config.name or "custom",
            )
            errors = [str(e)]
            result = None

        # Evaluate on test set if provided
        detailed_results = []
        eval_time = 0

        if result is not None and test_dataset is not None:
            eval_start = time.time()
            try:
                test_results = agent.evaluate_on_dataset(test_dataset, evaluator, verbose=verbose)
                eval_time = time.time() - eval_start

                metrics.test_score = test_results["average_score"]
                metrics.evaluation_time = eval_time
                detailed_results = test_results.get("detailed_results", [])

            except Exception as e:
                eval_time = time.time() - eval_start
                if verbose:
                    print(f"⚠️  Test evaluation failed: {e}")
                errors.append(f"Test evaluation error: {e}")

        # Set total time
        metrics.total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmark Complete!")
            print(f"  Total time: {metrics.total_time:.2f}s")
            print(f"  Train score: {metrics.train_score:.3f}")
            print(f"  Val score: {metrics.val_score:.3f}")
            if metrics.test_score is not None:
                print(f"  Test score: {metrics.test_score:.3f}")
            print(f"{'='*60}\n")

        # Create result
        benchmark_result = BenchmarkResult(
            config=config.to_dict(),
            metrics=metrics,
            detailed_results=detailed_results,
            errors=errors,
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def run_multiple_benchmarks(
        self,
        agent,
        configs: List[OptimizationConfig],
        evaluator: Optional[CleaningEvaluator] = None,
        test_dataset: Optional[CleaningDataset] = None,
        verbose: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for multiple configurations.

        Args:
            agent: DataCleaningAgent instance
            configs: List of configurations to benchmark
            evaluator: Custom evaluator (uses default if None)
            test_dataset: Optional test set for final evaluation
            verbose: Print progress information

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        for i, config in enumerate(configs):
            if verbose:
                print(f"\nRunning benchmark {i+1}/{len(configs)}")

            result = self.run_single_benchmark(
                agent=agent,
                config=config,
                evaluator=evaluator,
                test_dataset=test_dataset,
                verbose=verbose,
            )
            results.append(result)

            # Save intermediate results
            self.save_results()

        return results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save all benchmark results to file"""
        output_path = self.output_dir / filename

        all_results = {
            "benchmarks": [r.to_dict() for r in self.results],
            "summary": self.get_summary(),
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"✅ Saved benchmark results to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all benchmarks"""
        if not self.results:
            return {}

        summary = {
            "num_benchmarks": len(self.results),
            "optimizers": {},
        }

        # Group by optimizer
        by_optimizer: Dict[str, List[BenchmarkMetrics]] = {}
        for result in self.results:
            opt_name = result.metrics.optimizer_name
            if opt_name not in by_optimizer:
                by_optimizer[opt_name] = []
            by_optimizer[opt_name].append(result.metrics)

        # Calculate statistics per optimizer
        for opt_name, metrics_list in by_optimizer.items():
            summary["optimizers"][opt_name] = {
                "num_runs": len(metrics_list),
                "avg_train_score": sum(m.train_score for m in metrics_list) / len(metrics_list),
                "avg_val_score": sum(m.val_score for m in metrics_list) / len(metrics_list),
                "avg_time": sum(m.total_time for m in metrics_list) / len(metrics_list),
                "best_val_score": max(m.val_score for m in metrics_list),
                "worst_val_score": min(m.val_score for m in metrics_list),
            }

            # Add test scores if available
            test_scores = [m.test_score for m in metrics_list if m.test_score is not None]
            if test_scores:
                summary["optimizers"][opt_name]["avg_test_score"] = sum(test_scores) / len(
                    test_scores
                )
                summary["optimizers"][opt_name]["best_test_score"] = max(test_scores)

        return summary

    def create_comparison_table(self) -> pd.DataFrame:
        """Create a pandas DataFrame comparing all benchmark results"""
        data = []

        for result in self.results:
            m = result.metrics
            row = {
                "Optimizer": m.optimizer_name,
                "Config": m.config_name,
                "Train Score": f"{m.train_score:.3f}",
                "Val Score": f"{m.val_score:.3f}",
                "Test Score": f"{m.test_score:.3f}" if m.test_score else "N/A",
                "Total Time (s)": f"{m.total_time:.1f}",
                "Opt Time (s)": f"{m.optimization_time:.1f}",
                "# Examples": m.num_examples_used,
                "# Demos": m.num_demonstrations,
                "Timestamp": m.timestamp,
            }
            data.append(row)

        return pd.DataFrame(data)

    def print_comparison(self):
        """Print a formatted comparison table"""
        df = self.create_comparison_table()
        print("\n" + "=" * 100)
        print("BENCHMARK COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")

    @staticmethod
    def load_results(filepath: Path) -> "Benchmark":
        """Load benchmark results from a JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        benchmark = Benchmark(output_dir=filepath.parent)

        # Reconstruct BenchmarkResult objects
        for result_dict in data["benchmarks"]:
            metrics_dict = result_dict["metrics"]
            metrics = BenchmarkMetrics(
                total_time=metrics_dict["total_time"],
                optimization_time=metrics_dict["optimization_time"],
                evaluation_time=metrics_dict["evaluation_time"],
                train_score=metrics_dict["train_score"],
                val_score=metrics_dict["val_score"],
                test_score=metrics_dict.get("test_score"),
                num_examples_used=metrics_dict.get("num_examples_used", 0),
                num_demonstrations=metrics_dict.get("num_demonstrations", 0),
                num_iterations=metrics_dict.get("num_iterations", 0),
                memory_peak_mb=metrics_dict.get("memory_peak_mb"),
                num_llm_calls=metrics_dict.get("num_llm_calls"),
                optimizer_name=metrics_dict.get("optimizer_name", ""),
                config_name=metrics_dict.get("config_name", ""),
                timestamp=metrics_dict.get("timestamp", ""),
            )

            result = BenchmarkResult(
                config=result_dict["config"],
                metrics=metrics,
                detailed_results=result_dict.get("detailed_results", []),
                errors=result_dict.get("errors", []),
            )
            benchmark.results.append(result)

        return benchmark
