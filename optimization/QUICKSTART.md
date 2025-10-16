# Quick Start: DSPy Optimization

This guide will help you get started with optimizing your data cleaning agent.

## Phase 1: Core Infrastructure (Current Status)

Phase 1 provides the foundation for DSPy optimization. You can create and manage training datasets, evaluate cleaning plans, and configure optimization settings.

**Note**: Phase 2 (actual optimization execution) is coming next. For now, you can prepare your training data and configurations.

## Step 1: Create Training Examples

Create a JSON file with your training examples. You have two options:

### Option A: Full Plan Specification (Recommended for Best Quality)

```json
{
  "examples": [
    {
      "input_path": "data/messy_customers.csv",
      "expected_cleaning_plan": "1. remove_duplicates(subset=['customer_id', 'email'], keep='first')\n2. handle_missing_values(columns=['age', 'income'], numeric_strategy='median', categorical_strategy='most_frequent')\n3. clean_text_columns(columns=['email', 'name'], operations=['strip', 'lower'])\n4. convert_data_types(type_conversions={'age': 'int', 'income': 'float'})",
      "expected_rationale": "First eliminate duplicate customers based on ID and email. Then impute missing demographic data using median for numeric fields. Normalize text fields for consistency. Finally ensure proper data types for numeric columns.",
      "description": "Customer data cleaning with duplicates and missing values"
    }
  ]
}
```

### Option B: Structured Operations (Good Balance)

```json
{
  "examples": [
    {
      "input_path": "data/messy_customers.csv",
      "expected_operations": [
        {
          "operation": "remove_duplicates",
          "subset": ["customer_id", "email"],
          "keep": "first",
          "rationale": "Remove duplicate customer records based on ID and email"
        },
        {
          "operation": "handle_missing_values",
          "columns": ["age", "income"],
          "numeric_strategy": "median",
          "categorical_strategy": "most_frequent",
          "rationale": "Impute missing demographic data appropriately"
        },
        {
          "operation": "clean_text_columns",
          "columns": ["email", "name"],
          "operations": ["strip", "lower"],
          "rationale": "Normalize text data for consistency"
        }
      ],
      "description": "Customer data cleaning"
    }
  ]
}
```

**Tips for Creating Good Training Examples:**

1. **Be specific about columns**: Always specify which columns operations apply to
2. **Include parameters**: Specify strategies (median, most_frequent, etc.)
3. **Vary your examples**: Include different data quality issues
4. **Start with 5-10 examples**: More is better, but start small
5. **Use real data**: Examples from actual use cases work best

## Step 2: Load and Validate Your Dataset

```python
from optimization import CleaningDataset

# Load your dataset
dataset = CleaningDataset.from_json("my_training_examples.json")

# Validate it
errors = dataset.validate()
if errors:
    print("⚠️ Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✅ Dataset is valid!")

# Get summary
summary = dataset.summary()
print(f"📊 Dataset has {summary['total_examples']} examples")
print(f"   - Option A (full plans): {summary['specification_types']['plan']}")
print(f"   - Option B (structured ops): {summary['specification_types']['operations']}")
```

## Step 3: Test Your Evaluators

```python
from optimization import dspy_metric, create_default_evaluator
from file_io_tool import FileIOTool
import dspy

# Create metric
io_tool = FileIOTool()
metric = dspy_metric(io_tool)

# Test with an example
example = dspy.Example(
    comprehensive_data_analysis="Sample analysis...",
    overall_cleaning_plan="1. remove_duplicates(subset=['id'], keep='first')\n2. handle_missing_values(columns=['age'], numeric_strategy='median')",
    rationale="Remove duplicates then impute"
).with_inputs('comprehensive_data_analysis')

# Mock a prediction (in Phase 2, this will come from the agent)
class MockPrediction:
    overall_cleaning_plan = "1. remove_duplicates(subset=['id'], keep='first')\n2. handle_missing_values(columns=['age'], numeric_strategy='median')"

prediction = MockPrediction()

# Evaluate
score = metric(example, prediction)
print(f"📈 Evaluation score: {score:.2f}")
```

## Step 4: Create an Optimization Configuration

```python
from optimization import create_balanced_optimization_config

# Create a config
config = create_balanced_optimization_config(
    name="customer_data_optimization_v1",
    dataset_path="my_training_examples.json",
    description="Optimize for customer data cleaning"
)

# Save it for later
config.save_json("configs/customer_optimization.json")
config.save_yaml("configs/customer_optimization.yaml")

print(f"✅ Configuration saved!")
print(f"   Optimizer: {config.optimizer.optimizer_type}")
print(f"   Max demos: {config.optimizer.max_bootstrapped_demos}")
print(f"   Train ratio: {config.train_ratio}")
```

## Step 5: Prepare for Phase 2

Once Phase 2 is implemented, you'll be able to run optimization like this:

```python
# THIS WILL WORK IN PHASE 2 (coming soon!)
from data_cleaning_agent import DataCleaningAgent
from file_io_tool import FileIOTool
from stats_tool import StatisticalAnalysisTool
from data_transformation_tool import DataTransformationTool
from optimization import OptimizationConfig

# Load configuration
config = OptimizationConfig.load_json("configs/customer_optimization.json")

# Create agent
io_tool = FileIOTool()
stats_tool = StatisticalAnalysisTool()
transform_tool = DataTransformationTool()
agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)

# Optimize! (Phase 2)
# agent.compile_with_optimizer(config)

# Use optimized agent
# result = agent.clean_dataset("new_messy_data.csv")

# Save optimized model
# agent.save_compiled_model("models/optimized_cleaner_v1.json")
```

## Example Workflow

Here's a complete example workflow:

```python
# 1. Create training examples (do this once)
from optimization import CleaningExample, CleaningDataset

examples = [
    CleaningExample(
        input_path="data/example1.csv",
        expected_cleaning_plan="...",
        expected_rationale="...",
        description="Example 1"
    ),
    CleaningExample(
        input_path="data/example2.csv",
        expected_cleaning_plan="...",
        expected_rationale="...",
        description="Example 2"
    ),
    # Add more examples...
]

dataset = CleaningDataset(examples)
dataset.to_json("my_training_dataset.json")

# 2. Validate dataset
errors = dataset.validate()
if errors:
    print("Fix these errors:", errors)
    exit(1)

# 3. Create configuration
from optimization import create_balanced_optimization_config

config = create_balanced_optimization_config(
    name="my_optimization",
    dataset_path="my_training_dataset.json"
)
config.save_json("my_optimization_config.json")

# 4. Wait for Phase 2 to run actual optimization!
print("✅ Ready for optimization (Phase 2 coming soon)!")
```

## Tips for Success

### Creating Quality Training Examples

1. **Start with diverse examples**: Include different types of data quality issues
2. **Be explicit**: Specify columns, parameters, and operations clearly
3. **Test on representative data**: Use examples similar to your production data
4. **Iterate**: Start with a few examples, optimize, evaluate, add more

### How Many Examples?

- **Minimum**: 3-5 examples (for initial testing)
- **Recommended**: 10-15 examples (for good results)
- **Optimal**: 20+ examples (for production quality)

### Dataset Split

- **Default**: 80% train, 20% validation
- **Small datasets** (<10 examples): Use 100% for training
- **Large datasets** (20+ examples): Use 85/15 or 80/20 split

## Troubleshooting

### "Input file does not exist"
Make sure your `input_path` in examples points to actual CSV files. Use relative or absolute paths.

### "Could not load dataset"
Check your JSON syntax. Use a JSON validator if needed.

### "Must provide expected_cleaning_plan or expected_operations"
Each example needs at least one specification method (Option A or B).

## Next Steps

1. ✅ Create your training dataset
2. ✅ Validate it
3. ✅ Create optimization configuration
4. ⏳ Wait for Phase 2 (optimizer integration)
5. ⏳ Run optimization
6. ⏳ Evaluate results
7. ⏳ Deploy optimized agent

## Questions?

Check the full documentation in `optimization/README.md` or create an issue in the repository.
