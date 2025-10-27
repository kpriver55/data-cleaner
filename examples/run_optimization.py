"""
Example script demonstrating DSPy optimization

This script shows how to:
1. Load a training dataset
2. Configure optimization settings
3. Run optimization on the DataCleaningAgent
4. Evaluate and save the optimized model
5. Load and use the optimized model

Usage:
    python examples/run_optimization.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_cleaning_agent import DataCleaningAgent
from data_transformation_tool import DataTransformationTool
from file_io_tool import FileIOTool
from llm_config import setup_llm
from optimization import CleaningDataset, create_balanced_optimization_config
from stats_tool import StatisticalAnalysisTool


def main():
    print("=" * 70)
    print(" DSPy Optimization Example")
    print("=" * 70)

    # Step 1: Configure LLM
    print("\n📌 Step 1: Configuring LLM...")
    print("Using local Ollama model (make sure Ollama is running!)")

    try:
        setup_llm("ollama", "llama3.2:3b-instruct-q6_k")
        print("✅ LLM configured successfully")
    except Exception as e:
        print(f"❌ Failed to configure LLM: {e}")
        print("Make sure Ollama is running and the model is downloaded")
        print("  ollama pull llama3.2:3b-instruct-q6_k")
        return

    # Step 2: Create agent
    print("\n📌 Step 2: Creating DataCleaningAgent...")
    io_tool = FileIOTool()
    stats_tool = StatisticalAnalysisTool()
    transform_tool = DataTransformationTool()
    agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)
    print("✅ Agent created")

    # Step 3: Create optimization configuration
    print("\n📌 Step 3: Creating optimization configuration...")

    dataset_path = "examples/optimization_examples/training_dataset.json"

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Training dataset not found at: {dataset_path}")
        print("Please create a training dataset first.")
        print("See optimization/QUICKSTART.md for details.")
        return

    config = create_balanced_optimization_config(
        name="example_optimization",
        dataset_path=dataset_path,
        description="Example optimization run",
    )

    print(f"✅ Configuration created:")
    print(f"   Optimizer: {config.optimizer.optimizer_type}")
    print(f"   Max demos: {config.optimizer.max_bootstrapped_demos}")
    print(f"   Dataset: {dataset_path}")

    # Step 4: Run optimization
    print("\n📌 Step 4: Running optimization...")
    print("⚠️  This may take several minutes...")

    try:
        result = agent.compile_with_optimizer(config, verbose=True)
        print("\n✅ Optimization completed successfully!")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        print("Common issues:")
        print("  - LLM not responding (check Ollama)")
        print("  - Invalid training examples")
        print("  - Insufficient examples (need at least 1)")
        return

    # Step 5: Save optimized model
    print("\n📌 Step 5: Saving optimized model...")

    model_path = "models/optimized_agent.pkl"
    Path("models").mkdir(exist_ok=True)

    agent.save_compiled_model(
        model_path, metadata={"config": config.to_dict(), "result": result.to_dict()}
    )
    print(f"✅ Model saved to {model_path}")

    # Step 6: Test loading the model
    print("\n📌 Step 6: Testing model loading...")

    loaded_agent = DataCleaningAgent.load_compiled_model(
        model_path, io_tool, stats_tool, transform_tool
    )
    print("✅ Model loaded successfully")

    # Step 7: Evaluate on dataset (optional)
    print("\n📌 Step 7: Evaluating on training dataset...")
    dataset = CleaningDataset.from_json(dataset_path)

    try:
        eval_results = loaded_agent.evaluate_on_dataset(dataset, verbose=True)
        print(f"\n✅ Evaluation complete!")
        print(f"   Average score: {eval_results['average_score']:.4f}")
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        print("This is non-critical - the model was still optimized and saved")

    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"✅ Optimization completed in {result.duration_seconds:.2f} seconds")
    print(f"✅ Training score: {result.train_score:.4f}")
    if result.val_score is not None:
        print(f"✅ Validation score: {result.val_score:.4f}")
    print(f"✅ Model saved to: {model_path}")
    print("\nNext steps:")
    print("  1. Use the optimized agent to clean new datasets")
    print("  2. Compare performance with baseline agent")
    print("  3. Add more training examples and re-optimize")
    print("  4. Deploy to production")


if __name__ == "__main__":
    main()
