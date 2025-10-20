# CLI Usage Guide

Command-line interface for DSPy optimization tasks.

## Installation

```bash
# Optional: Install rich for better terminal output
pip install rich
```

## Commands

### 1. Create Dataset

Create a new empty training dataset:

```bash
python -m optimization.cli create-dataset --output my_dataset.json

# Or YAML format
python -m optimization.cli create-dataset --output my_dataset.yaml --format yaml

# Force overwrite existing file
python -m optimization.cli create-dataset --output my_dataset.json --force
```

### 2. Add Example

Add training examples to a dataset:

**Option A: Full Plan Specification (Recommended)**

```bash
# Create text files with plan and rationale
echo "1. remove_duplicates(subset=['id'], keep='first')
2. handle_missing_values(columns=['age'], numeric_strategy='median')" > plan.txt

echo "Remove duplicates first, then impute missing values" > rationale.txt

# Add to dataset
python -m optimization.cli add-example \
  --dataset my_dataset.json \
  --input data/messy.csv \
  --plan plan.txt \
  --rationale rationale.txt \
  --description "Clean customer data"
```

**Option B: Structured Operations**

```bash
# Create JSON with operations
cat > operations.json << 'EOF'
[
  {
    "operation": "remove_duplicates",
    "subset": ["id", "email"],
    "keep": "first",
    "rationale": "Remove duplicate records"
  },
  {
    "operation": "handle_missing_values",
    "columns": ["age", "salary"],
    "numeric_strategy": "median",
    "rationale": "Impute missing values"
  }
]
EOF

# Add to dataset
python -m optimization.cli add-example \
  --dataset my_dataset.json \
  --input data/messy.csv \
  --operations operations.json \
  --description "Clean customer data"
```

### 3. Validate Dataset

Check if dataset is valid:

```bash
python -m optimization.cli validate --dataset my_dataset.json
```

Output:
```
✓ Dataset is valid!

Dataset Summary:
  Total examples: 5
  Full plan specs: 3
  Structured ops: 2
  With descriptions: 5
```

### 4. Run Optimization

Run full optimization:

**Step 1: Create configuration file**

```json
{
  "name": "my_optimization",
  "dataset_path": "my_dataset.json",
  "output_dir": "optimization_results",
  "optimizer": {
    "optimizer_type": "BootstrapFewShot",
    "max_bootstrapped_demos": 4,
    "max_labeled_demos": 12,
    "num_threads": 8
  },
  "train_ratio": 0.8
}
```

Save as `config.json`

**Step 2: Run optimization**

```bash
python -m optimization.cli optimize \
  --config config.json \
  --output models/optimized_v1.pkl \
  --llm-provider ollama \
  --llm-model llama3.2:3b-instruct-q6_k
```

**With OpenAI:**

```bash
# Make sure OPENAI_API_KEY is set in environment
export OPENAI_API_KEY="sk-..."

python -m optimization.cli optimize \
  --config config.json \
  --output models/optimized_v1.pkl \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

**Quiet mode (less output):**

```bash
python -m optimization.cli optimize \
  --config config.json \
  --output models/optimized_v1.pkl \
  --quiet
```

### 5. Evaluate Model

Evaluate a saved model on test data:

```bash
python -m optimization.cli evaluate \
  --model models/optimized_v1.pkl \
  --test-dataset test_data.json \
  --output results.json
```

Output:
```
✓ Evaluation complete

Evaluation Results:
  Average Score: 0.8523
  Min Score: 0.7100
  Max Score: 0.9500
  Examples: 10

✓ Results saved to: results.json
```

## Complete Workflow Example

```bash
# 1. Create dataset
python -m optimization.cli create-dataset --output training.json

# 2. Add training examples
for file in data/messy_*.csv; do
  python -m optimization.cli add-example \
    --dataset training.json \
    --input "$file" \
    --operations "operations_${file##*/}.json" \
    --description "Clean $(basename $file)"
done

# 3. Validate
python -m optimization.cli validate --dataset training.json

# 4. Create config (use preset or custom JSON)
cat > opt_config.json << 'EOF'
{
  "name": "production_v1",
  "dataset_path": "training.json",
  "optimizer": {
    "optimizer_type": "BootstrapFewShot",
    "max_bootstrapped_demos": 4,
    "max_labeled_demos": 12
  },
  "train_ratio": 0.8
}
EOF

# 5. Run optimization
python -m optimization.cli optimize \
  --config opt_config.json \
  --output models/production_v1.pkl \
  --llm-provider ollama \
  --llm-model qwen2.5:7b-instruct-q5_k_m

# 6. Evaluate on test set
python -m optimization.cli evaluate \
  --model models/production_v1.pkl \
  --test-dataset test.json \
  --output eval_results.json
```

## Tips

### Creating Quality Training Examples

1. **Be specific about columns**: Always specify which columns operations apply to
2. **Include parameters**: Specify strategies (median, most_frequent, etc.)
3. **Vary examples**: Include different data quality issues
4. **Start with 5-10 examples**: More is better, but start small
5. **Use real data**: Examples from actual use cases work best

### Choosing Optimizer Settings

**Quick (Fast, for testing)**
- Optimizer: `BootstrapFewShot`
- Max demos: 3
- Labeled demos: 8
- Use when: Testing, small datasets

**Balanced (Recommended)**
- Optimizer: `BootstrapFewShot`
- Max demos: 4
- Labeled demos: 12
- Use when: Most production use cases

**Thorough (Best quality)**
- Optimizer: `MIPRO`
- Max demos: 8
- Labeled demos: 16
- Candidate programs: 20
- Use when: Production models, maximum quality needed

### Environment Variables

Set these for API-based LLM providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Together AI
export TOGETHER_API_KEY="..."

# Anyscale
export ANYSCALE_API_KEY="..."
```

Or use a `.env` file (see `.env.example`).

## Troubleshooting

### "Dataset not found"
- Check the path is correct
- Use absolute paths if relative paths don't work

### "API key not found"
- Set environment variable for your provider
- Check `.env` file if using one

### "Optimization failed"
- Check LLM is responding (test with simple query)
- Verify dataset is valid with `validate` command
- Check you have enough examples (minimum 1, recommended 5+)

### "Error loading model"
- Ensure model file exists and isn't corrupted
- Check it was created with same DSPy version

## Advanced Usage

### Custom Evaluators

Create a Python script with custom evaluation logic:

```python
from optimization import CleaningDataset, dspy_metric
from optimization.evaluators import CompositeEvaluator, OperationPresenceEvaluator

# Custom weights
evaluator = CompositeEvaluator(io_tool, [
    (OperationPresenceEvaluator(io_tool), 0.7),
    # Add more evaluators...
])

metric = dspy_metric(io_tool, evaluator=evaluator)

# Use in optimization...
```

### Batch Processing

Process multiple datasets:

```bash
#!/bin/bash
for dataset in datasets/*.json; do
  name=$(basename "$dataset" .json)
  python -m optimization.cli optimize \
    --config "configs/${name}_config.json" \
    --output "models/${name}.pkl"
done
```

## See Also

- [Main README](../README.md) - Project overview
- [Optimization README](README.md) - Technical documentation
- [QUICKSTART](QUICKSTART.md) - Quick start guide
- [PROJECT_OUTLINE](PROJECT_OUTLINE.md) - Full project plan
