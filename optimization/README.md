# DSPy Optimization Module

This module provides tools for optimizing the data cleaning agent using DSPy's optimization capabilities.

## Overview

The optimization module allows you to improve the data cleaning agent's performance by training it on examples of your data cleaning tasks. Instead of using hardcoded prompts, you can optimize the agent to produce better cleaning plans based on your specific use cases.

**Key Features:**

- 📊 **Dataset Management**: Create and manage training examples
- 🎯 **Multiple Evaluators**: Assess plan quality from different angles
- ⚙️ **Flexible Configuration**: Preset and custom optimization settings
- 🚀 **Multiple Optimizers**: BootstrapFewShot, MIPRO, and more
- 📈 **Benchmarking Tools**: Compare configurations and measure performance
- 💻 **CLI & Web UI**: Use command-line or Streamlit interface

## Phase 1: Core Infrastructure (✅ Completed)

Phase 1 provides the foundation for DSPy optimization:

- **Dataset Management** (`dataset.py`): Load, manage, and convert training examples
- **Evaluation Metrics** (`evaluators.py`): Assess cleaning plan quality
- **Configuration** (`config.py`): Manage optimization settings
- **Unit Tests** (`test_evaluators.py`): Verify evaluator correctness

## Components

### 1. Dataset Management (`dataset.py`)

Manages training and test examples for optimization.

#### CleaningExample

Represents a single training example. Supports three specification levels:

**Option A: Full Plan Specification (Best Quality)**

```python
from optimization import CleaningExample

example = CleaningExample(
    input_path="data/messy.csv",
    expected_cleaning_plan="""1. remove_duplicates(subset=['id'], keep='first')
2. handle_missing_values(columns=['age', 'salary'], numeric_strategy='median')
3. clean_text_columns(columns=['email'], operations=['strip', 'lower'])""",
    expected_rationale="Remove duplicates first, then impute missing values, finally normalize text",
    description="Clean customer data"
)
```

**Option B: Structured Operations (Good Quality)**

```python
example = CleaningExample(
    input_path="data/messy.csv",
    expected_operations=[
        {
            "operation": "remove_duplicates",
            "columns": ["id", "email"],
            "keep": "first",
            "rationale": "Remove duplicate customer records"
        },
        {
            "operation": "handle_missing_values",
            "columns": ["age", "salary"],
            "numeric_strategy": "median",
            "rationale": "Impute with median to preserve distribution"
        }
    ],
    description="Clean customer data"
)
```

**Option C: Input/Output Comparison (Easiest - Auto-inferred)**

```python
# Automatically infer plan by comparing raw and cleaned data
example = CleaningExample(
    input_path="data/raw.csv",
    expected_output_path="data/cleaned.csv",
    description="Clean customer data"
)

# Or use the convenience function
from optimization import infer_plan_from_files

result = infer_plan_from_files("data/raw.csv", "data/cleaned.csv")
example = CleaningExample(input_path="data/raw.csv", **result)
```

**Note**: Option C uses an LLM to automatically infer the cleaning steps by analyzing differences between raw and cleaned datasets. This is the easiest approach (just provide before/after files) but may produce less accurate plans than manually specified options A or B. It's great for getting started quickly!

#### CleaningDataset

Manages collections of examples:

```python
from optimization import CleaningDataset

# Load from JSON
dataset = CleaningDataset.from_json("examples/training_dataset.json")

# Validate examples
errors = dataset.validate()
if errors:
    print("Validation errors:", errors)

# Split into train/test
train_dataset, test_dataset = dataset.split(train_ratio=0.8)

# Get summary
print(dataset.summary())

# Convert to DSPy examples
from file_io_tool import FileIOTool
io_tool = FileIOTool()
dspy_examples = dataset.to_dspy_examples(io_tool)
```

### 2. Evaluation Metrics (`evaluators.py`)

Provides metrics to assess cleaning plan quality. Focuses on what matters: operations, parameters, and column specificity.

#### Available Evaluators

1. **OperationPresenceEvaluator**: Checks if all expected operations are present
2. **ParameterAccuracyEvaluator**: Verifies correct parameters (strategies, methods, thresholds)
3. **ColumnSpecificityEvaluator**: Ensures plans specify which columns to operate on
4. **PlanStructureEvaluator**: Validates plan formatting and structure
5. **CompositeEvaluator**: Combines multiple evaluators with weights

#### Using Evaluators

```python
from optimization import dspy_metric, binary_metric, create_default_evaluator
from file_io_tool import FileIOTool

io_tool = FileIOTool()

# Create default metric (composite of all evaluators)
metric = dspy_metric(io_tool)

# Or create custom evaluator
from optimization import CompositeEvaluator, OperationPresenceEvaluator, ColumnSpecificityEvaluator

custom_evaluator = CompositeEvaluator(io_tool, [
    (OperationPresenceEvaluator(io_tool), 0.6),
    (ColumnSpecificityEvaluator(io_tool), 0.4)
])
metric = dspy_metric(io_tool, evaluator=custom_evaluator)

# Binary metric for pass/fail evaluation
binary = binary_metric(io_tool, threshold=0.75)
```

### 3. Configuration (`config.py`)

Manages optimization settings and hyperparameters.

#### Configuration Classes

- **OptimizerConfig**: Optimizer-specific settings (type, num demos, threads, etc.)
- **EvaluationConfig**: Evaluation settings (evaluator type, weights, binary threshold)
- **OptimizationConfig**: Complete optimization run configuration

#### Creating Configurations

```python
from optimization import (
    OptimizationConfig,
    create_quick_optimization_config,
    create_balanced_optimization_config,
    create_thorough_optimization_config
)

# Quick optimization (fast, for testing)
config = create_quick_optimization_config(
    name="quick_test",
    dataset_path="examples/training_dataset.json"
)

# Balanced optimization (recommended)
config = create_balanced_optimization_config(
    name="production_v1",
    dataset_path="examples/training_dataset.json"
)

# Thorough optimization (slow, best quality)
config = create_thorough_optimization_config(
    name="production_v2",
    dataset_path="examples/training_dataset.json"
)

# Custom configuration
from optimization import OptimizerConfig, EvaluationConfig

config = OptimizationConfig(
    name="custom_optimization",
    dataset_path="examples/training_dataset.json",
    output_dir="optimization_results",
    optimizer=OptimizerConfig(
        optimizer_type="BootstrapFewShot",
        max_bootstrapped_demos=4,
        max_labeled_demos=12,
        num_threads=8
    ),
    evaluation=EvaluationConfig(
        evaluator_type="default",
        use_binary_metric=False
    ),
    train_ratio=0.8,
    random_seed=42
)

# Save configuration
config.save_json("configs/my_optimization.json")
config.save_yaml("configs/my_optimization.yaml")

# Load configuration
config = OptimizationConfig.load_json("configs/my_optimization.json")
```

## Example Training Dataset Format

See `examples/optimization_examples/training_dataset.json` for a complete example.

### JSON Format (Option A - Full Plan)

```json
{
  "examples": [
    {
      "input_path": "data/messy.csv",
      "expected_cleaning_plan": "1. remove_duplicates(subset=['id'], keep='first')\n2. handle_missing_values(columns=['age', 'salary'], numeric_strategy='median')\n3. clean_text_columns(columns=['email'], operations=['strip', 'lower'])",
      "expected_rationale": "Remove duplicates first to ensure uniqueness. Impute missing values with median. Normalize text data.",
      "description": "Clean customer data"
    }
  ]
}
```

### JSON Format (Option B - Structured Operations)

```json
{
  "examples": [
    {
      "input_path": "data/messy.csv",
      "expected_operations": [
        {
          "operation": "remove_duplicates",
          "columns": ["id", "email"],
          "keep": "first",
          "rationale": "Remove duplicate customer records"
        },
        {
          "operation": "handle_missing_values",
          "columns": ["age", "salary"],
          "numeric_strategy": "median",
          "categorical_strategy": "most_frequent",
          "rationale": "Impute missing values appropriately"
        }
      ],
      "description": "Clean customer data"
    }
  ]
}
```

## Running Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
python -m pytest optimization/test_evaluators.py -v

# Run specific test
python -m pytest optimization/test_evaluators.py::TestOperationPresenceEvaluator -v
```

## Next Steps: Phase 2

Phase 2 will add the actual optimization functionality:

- **Optimizer Wrappers** (`optimizers.py`): Wrappers for DSPy optimizers (BootstrapFewShot, MIPRO, etc.)
- **Agent Integration**: Extend `DataCleaningAgent` with optimization methods:
  - `compile_with_optimizer()`
  - `evaluate()`
  - `save_compiled_model()` / `load_compiled_model()`
- **CLI Tool**: Command-line interface for running optimizations
- **Streamlit Integration**: UI for optimization workflow

## Design Decisions

### Why Focus on Plan Quality, Not Rationale?

The evaluation metrics focus on the cleaning plan (operations, parameters, columns) rather than the rationale explanation because:

1. **Plan determines output**: The plan is what gets executed; rationale is informational only
2. **High variance**: Rationale text is subjective and highly variable
3. **Optimization risk**: Optimizing for rationale similarity could waste effort on text matching rather than improving decisions
4. **User value**: Users care about clean data, not matching explanation styles

### Three-Tier Example Specification

We support three levels of specification (A, B, C) to give users flexibility:

- **Option A (Full Plan)**: Best quality, most control, but requires manual effort
- **Option B (Structured Ops)**: Good balance of quality and ease of creation
- **Option C (Input/Output)**: Easiest to create using auto-diff, but may be less accurate

**How Option C Works:**

Option C uses a simple LLM-based approach to infer cleaning operations:
1. Loads both raw and cleaned datasets
2. Summarizes each dataset (shape, columns, types, sample data, quality issues)
3. Asks an LLM to infer what operations were performed
4. Returns the inferred plan in the expected format

**Limitations**: This is a starting point that focuses on functionality over accuracy. The LLM may:
- Miss subtle transformations
- Infer incorrect parameter values
- Get operation order wrong
- Struggle with complex multi-step cleanings

Despite these limitations, Option C is valuable for:
- Quickly bootstrapping a training dataset
- Cases where you have before/after examples but no documentation
- Iterative refinement (start with Option C, manually improve to A/B)

## File Structure

```
optimization/
├── __init__.py              # Module exports
├── dataset.py               # Training example management
├── evaluators.py            # Evaluation metrics
├── config.py                # Configuration management
├── optimizers.py            # Optimizer wrappers
├── plan_inference.py        # Auto-diff plan inference (Option C)
├── benchmarking.py          # Performance benchmarking tools
├── benchmark_analysis.py    # Analysis and visualization tools
├── test_evaluators.py       # Unit tests
├── cli.py                   # Command-line interface
├── README.md                # This file
├── QUICKSTART.md            # Quick start guide
└── CLI_GUIDE.md             # CLI usage guide
```

## Benchmarking

Compare different optimization configurations to find the best setup for your use case.

### Quick Start

```bash
# Run benchmark comparing multiple configurations
python examples/run_benchmarks.py

# Analyze results
python optimization/benchmark_analysis.py benchmarks/benchmark_results.json
```

### Programmatic Usage

```python
from optimization import Benchmark, OptimizationConfig

# Create benchmark
benchmark = Benchmark(output_dir=Path("benchmarks"))

# Define configurations to compare
configs = [
    OptimizationConfig(name="quick", max_bootstrapped_demos=2),
    OptimizationConfig(name="balanced", max_bootstrapped_demos=4),
    OptimizationConfig(name="thorough", max_bootstrapped_demos=8),
]

# Run benchmarks
results = benchmark.run_multiple_benchmarks(
    agent=agent,
    configs=configs,
    test_dataset=test_dataset
)

# View comparison
benchmark.print_comparison()
```

See [Benchmarking Guide](../docs/BENCHMARKING.md) for detailed instructions.

## Contributing

When adding new evaluators:

1. Inherit from `CleaningEvaluator`
2. Implement `evaluate(example, prediction, trace)` returning float in [0, 1]
3. Add unit tests in `test_evaluators.py`
4. Update `create_default_evaluator()` if appropriate
5. Document in this README

## Questions?

For issues or questions, please create an issue in the repository.
