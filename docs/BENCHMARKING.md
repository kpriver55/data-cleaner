# Benchmarking Guide

This guide explains how to benchmark different DSPy optimization strategies to find the best configuration for your use case.

## Overview

Benchmarking helps you:

- **Compare optimizers** (BootstrapFewShot, MIPRO, etc.)
- **Tune hyperparameters** (number of demos, training examples, etc.)
- **Measure performance** (accuracy, speed, resource usage)
- **Make informed decisions** about production configurations

## Quick Start

### 1. Prepare Your Dataset

Split your dataset into train/validation/test sets:

```python
from optimization.dataset import CleaningDataset

# Load full dataset
dataset = CleaningDataset.from_json("my_data.json")

# Split: 60% train, 20% val, 20% test
train_val, test = dataset.train_test_split(test_size=0.2, random_state=42)
train, val = train_val.train_test_split(test_size=0.25, random_state=42)

# Save splits
train.to_json("train.json")
val.to_json("val.json")
test.to_json("test.json")
```

### 2. Run Benchmark

```bash
python examples/run_benchmarks.py
```

This will:

1. Load your dataset splits
2. Run multiple optimization configurations
3. Evaluate each on the test set
4. Generate comparison reports

### 3. Analyze Results

```bash
# View quick summary
python optimization/benchmark_analysis.py benchmarks/benchmark_results.json

# Generate detailed report and plots
python optimization/benchmark_analysis.py benchmarks/benchmark_results.json
```

## Customizing Benchmarks

### Define Custom Configurations

Edit `examples/run_benchmarks.py` to test different configurations:

```python
configs = [
    # Fast baseline
    OptimizationConfig(
        name="baseline",
        optimizer_name="bootstrap_few_shot",
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    ),

    # Increase demos
    OptimizationConfig(
        name="more_demos",
        optimizer_name="bootstrap_few_shot",
        max_bootstrapped_demos=8,
        max_labeled_demos=8,
    ),

    # Try different optimizer
    OptimizationConfig(
        name="mipro",
        optimizer_name="mipro",
        max_bootstrapped_demos=4,
    ),
]
```

### Benchmark Different Aspects

**1. Number of Demonstrations**

Test how many examples the model needs:

```python
for n_demos in [2, 4, 6, 8, 10]:
    configs.append(OptimizationConfig(
        name=f"demos_{n_demos}",
        max_bootstrapped_demos=n_demos,
        max_labeled_demos=n_demos,
    ))
```

**2. Optimizer Comparison**

Compare different optimization strategies:

```python
optimizers = ["bootstrap_few_shot", "bootstrap_few_shot_with_random_search", "mipro"]

for opt_name in optimizers:
    configs.append(OptimizationConfig(
        name=opt_name,
        optimizer_name=opt_name,
    ))
```

**3. Dataset Size**

Test how much training data is needed:

```python
for size in [0.25, 0.5, 0.75, 1.0]:
    # Use subset of training data
    subset = train.sample(frac=size, random_state=42)
    subset.to_json(f"train_{int(size*100)}.json")

    configs.append(OptimizationConfig(
        name=f"data_{int(size*100)}pct",
        dataset_path=f"train_{int(size*100)}.json",
    ))
```

## Benchmark Metrics

### Quality Metrics

- **Train Score**: Performance on training set (should be high)
- **Val Score**: Performance on validation set (used for model selection)
- **Test Score**: Final performance on held-out test set (true performance)

### Timing Metrics

- **Total Time**: Wall-clock time for entire optimization
- **Optimization Time**: Time spent in optimizer
- **Evaluation Time**: Time spent evaluating on test set

### Resource Metrics

- **Examples Used**: Number of training examples used
- **Demonstrations**: Number of demos in final model
- **Iterations**: Number of optimizer iterations

## Interpreting Results

### Good Signs

✅ **Val score close to train score** - Model generalizes well
✅ **Test score close to val score** - No overfitting to validation set
✅ **Higher scores with more demos** - Model is learning effectively
✅ **Consistent results across runs** - Stable configuration

### Warning Signs

⚠️ **Val score much lower than train** - Overfitting, need more data or regularization
⚠️ **Test score much lower than val** - Overfitting to validation set
⚠️ **No improvement with more demos** - May need better examples or different optimizer
⚠️ **High variance across runs** - Unstable, try different random seed or more data

## Best Practices

### Dataset Preparation

1. **Use representative examples**: Cover diverse data quality issues
2. **Balance difficulty**: Mix easy and hard cases
3. **Sufficient size**: At least 20-30 examples recommended
4. **Clean labels**: Ensure expected outputs are high quality

### Configuration Selection

1. **Start small**: Begin with quick config to validate setup
2. **Iterate**: Gradually increase complexity (more demos, iterations)
3. **Test systematically**: Change one variable at a time
4. **Track everything**: Save all configurations and results

### Evaluation Strategy

1. **Always use test set**: Don't evaluate on training data
2. **Multiple metrics**: Consider both quality and speed
3. **Statistical significance**: Run multiple seeds if results are close
4. **Real-world validation**: Test on actual production data

## Example Workflow

### Scenario: Finding Optimal Configuration

**Goal**: Maximize accuracy while keeping training time under 10 minutes

**Step 1: Baseline**

```python
# Quick configuration (~2 min)
baseline = OptimizationConfig(name="baseline", max_bootstrapped_demos=2)
# Result: Val=0.65, Time=1.5min
```

**Step 2: Increase Demos**

```python
# More demonstrations (~5 min)
more_demos = OptimizationConfig(name="more_demos", max_bootstrapped_demos=6)
# Result: Val=0.78, Time=4.2min
```

**Step 3: Try Advanced Optimizer**

```python
# MIPRO (~8 min)
mipro = OptimizationConfig(name="mipro", optimizer_name="mipro")
# Result: Val=0.82, Time=7.8min
```

**Step 4: Fine-tune Winner**

```python
# Optimize MIPRO settings (~9 min)
tuned = OptimizationConfig(
    name="tuned",
    optimizer_name="mipro",
    max_bootstrapped_demos=8,
)
# Result: Val=0.85, Time=9.2min ✅ Winner!
```

## Programmatic Usage

### Running Benchmarks in Code

```python
from optimization.benchmarking import Benchmark
from optimization.config import OptimizationConfig

# Create benchmark harness
benchmark = Benchmark(output_dir=Path("benchmarks"))

# Define configs
configs = [...]

# Run benchmarks
results = benchmark.run_multiple_benchmarks(
    agent=agent,
    configs=configs,
    test_dataset=test_dataset,
    verbose=True
)

# Save results
benchmark.save_results("my_benchmark.json")

# Analyze
benchmark.print_comparison()
summary = benchmark.get_summary()
```

### Analyzing Results in Code

```python
from optimization.benchmark_analysis import (
    load_benchmark_results,
    compare_benchmarks,
    create_performance_plot,
    create_detailed_report
)

# Load results
data = load_benchmark_results("benchmarks/benchmark_results.json")

# Compare multiple runs
df = compare_benchmarks([
    "benchmarks/run1.json",
    "benchmarks/run2.json"
])

# Create visualizations
create_performance_plot(df, "benchmarks/plot.html")

# Generate report
report = create_detailed_report("benchmarks/results.json", "benchmarks/report.md")
```

## Troubleshooting

### Benchmarks Taking Too Long

**Problem**: Benchmark runs exceed time budget

**Solutions**:

- Reduce `max_bootstrapped_demos` and `max_labeled_demos`
- Use smaller validation set
- Use faster LLM (e.g., GPT-3.5 instead of GPT-4)
- Comment out expensive optimizers (MIPRO)

### Low Scores Across All Configs

**Problem**: All configurations achieve low validation scores

**Solutions**:

- Check dataset quality - ensure examples are correct
- Verify evaluator is appropriate for your task
- Try more training examples
- Consider if task is too difficult for current approach

### High Variance in Results

**Problem**: Results vary significantly between runs

**Solutions**:

- Set `random_seed` in configurations
- Increase dataset size
- Use more demonstrations
- Average results across multiple seeds

### Out of Memory Errors

**Problem**: Benchmark crashes with memory errors

**Solutions**:

- Reduce batch size in optimizer settings
- Use smaller dataset
- Clear cached models between runs
- Increase system memory or use cloud instance

## Advanced Topics

### Custom Evaluators

Use domain-specific evaluation metrics:

```python
from optimization.evaluators import CleaningEvaluator

class MyCustomEvaluator(CleaningEvaluator):
    def evaluate(self, example, prediction):
        # Your custom logic
        score = calculate_my_metric(example, prediction)
        return score

# Use in benchmark
benchmark.run_multiple_benchmarks(
    agent=agent,
    configs=configs,
    evaluator=MyCustomEvaluator(io_tool),
    test_dataset=test_dataset
)
```

### Parallel Benchmarking

Run multiple benchmarks in parallel:

```python
from multiprocessing import Pool

def run_single(config):
    # Create fresh agent instance
    agent = create_agent()
    benchmark = Benchmark()
    return benchmark.run_single_benchmark(agent, config)

# Run in parallel
with Pool(processes=4) as pool:
    results = pool.map(run_single, configs)
```

### Continuous Benchmarking

Integrate with CI/CD:

```yaml
# .github/workflows/benchmark.yml
name: Benchmark
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: python examples/run_benchmarks.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/
```

## Resources

- [DSPy Optimization Documentation](https://github.com/stanfordnlp/dspy)
- [Optimization Configuration Guide](../optimization/README.md)
- [Example Benchmark Script](../examples/run_benchmarks.py)
- [Benchmark Analysis Tools](../optimization/benchmark_analysis.py)

## Feedback

Found an issue or have suggestions for improving benchmarking? Please open an issue on GitHub.
