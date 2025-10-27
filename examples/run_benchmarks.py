"""
Example script demonstrating how to run benchmarks comparing different optimizers.

This script:
1. Loads a training dataset
2. Defines multiple optimization configurations to compare
3. Runs benchmarks for each configuration
4. Generates comparison reports and visualizations
"""

import sys
from pathlib import Path

import dspy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformation_tool import DataTransformationTool

from data_cleaning_agent import DataCleaningAgent
from file_io_tool import FileIOTool
from llm_config import LLMConfig
from optimization.benchmarking import Benchmark
from optimization.config import OptimizationConfig
from optimization.dataset import CleaningDataset
from stats_tool import StatsTool


def main():
    """Run benchmark comparison"""

    print("\n" + "=" * 80)
    print("DSPy Data Cleaning Agent - Benchmark Runner")
    print("=" * 80 + "\n")

    # ============================================================================
    # Step 1: Configure LLM
    # ============================================================================

    print("Step 1: Configuring LLM...")
    llm_config = LLMConfig()
    lm = llm_config.get_lm()
    dspy.settings.configure(lm=lm)
    print(f"✅ Configured: {llm_config.provider.value}")

    # ============================================================================
    # Step 2: Load Dataset
    # ============================================================================

    print("\nStep 2: Loading dataset...")

    # Replace with your actual dataset path
    dataset_path = Path("training_data/my_dataset.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nTo run benchmarks, you need a training dataset.")
        print("Create one using:")
        print("  python -m optimization.cli create-dataset -o training_data/my_dataset.json")
        print("\nOr use the Streamlit UI: Optimization > Dataset Manager")
        return 1

    dataset = CleaningDataset.from_json(dataset_path)
    print(f"✅ Loaded dataset with {len(dataset)} examples")

    # Split into train/val/test
    # For benchmarking, we typically use: 60% train, 20% val, 20% test
    train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_val_dataset.train_test_split(
        test_size=0.25, random_state=42
    )  # 0.25 * 0.8 = 0.2

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")

    # ============================================================================
    # Step 3: Initialize Agent
    # ============================================================================

    print("\nStep 3: Initializing agent...")
    io_tool = FileIOTool()
    stats_tool = StatsTool()
    transform_tool = DataTransformationTool()
    agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)
    print("✅ Agent initialized")

    # ============================================================================
    # Step 4: Define Benchmark Configurations
    # ============================================================================

    print("\nStep 4: Defining benchmark configurations...")

    # Save train/val splits for benchmarking
    train_path = Path("training_data/benchmark_train.json")
    val_path = Path("training_data/benchmark_val.json")
    test_path = Path("training_data/benchmark_test.json")

    train_dataset.to_json(train_path)
    val_dataset.to_json(val_path)
    test_dataset.to_json(test_path)

    # Define configurations to benchmark
    configs = [
        # Quick configuration for fast iteration
        OptimizationConfig(
            name="quick",
            dataset_path=str(train_path),
            val_dataset_path=str(val_path),
            optimizer_name="bootstrap_few_shot",
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            train_test_split=0.0,  # Already split
            random_seed=42,
        ),
        # Balanced configuration (recommended)
        OptimizationConfig(
            name="balanced",
            dataset_path=str(train_path),
            val_dataset_path=str(val_path),
            optimizer_name="bootstrap_few_shot",
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            train_test_split=0.0,
            random_seed=42,
        ),
        # Thorough configuration for best quality
        OptimizationConfig(
            name="thorough",
            dataset_path=str(train_path),
            val_dataset_path=str(val_path),
            optimizer_name="bootstrap_few_shot",
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            train_test_split=0.0,
            random_seed=42,
        ),
        # MIPRO optimizer (more expensive but potentially better)
        # OptimizationConfig(
        #     name="mipro_balanced",
        #     dataset_path=str(train_path),
        #     val_dataset_path=str(val_path),
        #     optimizer_name="mipro",
        #     max_bootstrapped_demos=4,
        #     max_labeled_demos=4,
        #     train_test_split=0.0,
        #     random_seed=42,
        # ),
    ]

    print(f"✅ Defined {len(configs)} configurations to benchmark")

    # ============================================================================
    # Step 5: Run Benchmarks
    # ============================================================================

    print("\nStep 5: Running benchmarks...")
    print("This may take a while depending on your dataset size and LLM speed.\n")

    benchmark = Benchmark(output_dir=Path("benchmarks"))

    results = benchmark.run_multiple_benchmarks(
        agent=agent,
        configs=configs,
        evaluator=None,  # Use default evaluator
        test_dataset=test_dataset,
        verbose=True,
    )

    # ============================================================================
    # Step 6: Display Results
    # ============================================================================

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80 + "\n")

    # Print comparison table
    benchmark.print_comparison()

    # Print summary statistics
    summary = benchmark.get_summary()
    print("\nSUMMARY STATISTICS:")
    print("-" * 80)
    for optimizer, stats in summary["optimizers"].items():
        print(f"\n{optimizer}:")
        print(f"  Runs: {stats['num_runs']}")
        print(f"  Avg Train Score: {stats['avg_train_score']:.3f}")
        print(f"  Avg Val Score: {stats['avg_val_score']:.3f}")
        if "avg_test_score" in stats:
            print(f"  Avg Test Score: {stats['avg_test_score']:.3f}")
        print(f"  Avg Time: {stats['avg_time']:.1f}s")
        print(f"  Best Val Score: {stats['best_val_score']:.3f}")

    # ============================================================================
    # Step 7: Identify Best Configuration
    # ============================================================================

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80 + "\n")

    # Find best by validation score
    best_result = max(results, key=lambda r: r.metrics.val_score)
    print(f"🏆 Best Configuration (by validation score):")
    print(f"  Name: {best_result.metrics.config_name}")
    print(f"  Optimizer: {best_result.metrics.optimizer_name}")
    print(f"  Val Score: {best_result.metrics.val_score:.3f}")
    if best_result.metrics.test_score:
        print(f"  Test Score: {best_result.metrics.test_score:.3f}")
    print(f"  Time: {best_result.metrics.total_time:.1f}s")

    # Find best balance (score / time)
    best_efficiency = max(results, key=lambda r: r.metrics.val_score / max(r.metrics.total_time, 1))
    print(f"\n⚡ Most Efficient Configuration (score/time):")
    print(f"  Name: {best_efficiency.metrics.config_name}")
    print(f"  Optimizer: {best_efficiency.metrics.optimizer_name}")
    print(f"  Val Score: {best_efficiency.metrics.val_score:.3f}")
    print(f"  Time: {best_efficiency.metrics.total_time:.1f}s")
    print(
        f"  Efficiency: {best_efficiency.metrics.val_score / max(best_efficiency.metrics.total_time, 1):.4f}"
    )

    print("\n" + "=" * 80)
    print(f"✅ Benchmark results saved to: benchmarks/benchmark_results.json")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
