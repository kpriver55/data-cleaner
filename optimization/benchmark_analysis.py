"""
Tools for analyzing and visualizing benchmark results.

Provides functions to:
- Load and compare multiple benchmark runs
- Generate performance plots
- Create detailed analysis reports
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def load_benchmark_results(filepath: Path) -> Dict:
    """Load benchmark results from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def compare_benchmarks(filepaths: List[Path]) -> pd.DataFrame:
    """
    Compare results from multiple benchmark runs.

    Args:
        filepaths: List of paths to benchmark result JSON files

    Returns:
        DataFrame with comparison across all runs
    """
    all_data = []

    for filepath in filepaths:
        data = load_benchmark_results(filepath)
        run_name = filepath.stem

        for benchmark in data["benchmarks"]:
            metrics = benchmark["metrics"]
            row = {
                "Run": run_name,
                "Optimizer": metrics.get("optimizer_name", "unknown"),
                "Config": metrics.get("config_name", "unknown"),
                "Train Score": metrics["train_score"],
                "Val Score": metrics["val_score"],
                "Test Score": metrics.get("test_score"),
                "Total Time": metrics["total_time"],
                "Opt Time": metrics["optimization_time"],
                "Eval Time": metrics["evaluation_time"],
                "Examples": metrics.get("num_examples_used", 0),
                "Demos": metrics.get("num_demonstrations", 0),
                "Timestamp": metrics.get("timestamp", ""),
            }
            all_data.append(row)

    return pd.DataFrame(all_data)


def create_performance_plot(df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Create interactive performance comparison plot.

    Args:
        df: DataFrame from compare_benchmarks()
        output_path: Optional path to save HTML plot
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️  Plotly not available. Install with: pip install plotly")
        return

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Validation Scores by Optimizer",
            "Training Time Comparison",
            "Score vs Time Trade-off",
            "Test Scores (if available)",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "bar"}]],
    )

    # Get unique optimizers and configs
    optimizers = df["Optimizer"].unique()
    configs = df["Config"].unique()

    # Color map
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Plot 1: Val Scores by Optimizer
    for i, optimizer in enumerate(optimizers):
        optimizer_data = df[df["Optimizer"] == optimizer]
        fig.add_trace(
            go.Bar(
                name=optimizer,
                x=optimizer_data["Config"],
                y=optimizer_data["Val Score"],
                marker_color=colors[i % len(colors)],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Plot 2: Time Comparison
    for i, optimizer in enumerate(optimizers):
        optimizer_data = df[df["Optimizer"] == optimizer]
        fig.add_trace(
            go.Bar(
                name=optimizer,
                x=optimizer_data["Config"],
                y=optimizer_data["Total Time"],
                marker_color=colors[i % len(colors)],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Plot 3: Score vs Time Scatter
    for i, optimizer in enumerate(optimizers):
        optimizer_data = df[df["Optimizer"] == optimizer]
        fig.add_trace(
            go.Scatter(
                name=optimizer,
                x=optimizer_data["Total Time"],
                y=optimizer_data["Val Score"],
                mode="markers+text",
                text=optimizer_data["Config"],
                textposition="top center",
                marker=dict(size=12, color=colors[i % len(colors)]),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Plot 4: Test Scores (if available)
    test_data = df[df["Test Score"].notna()]
    if not test_data.empty:
        for i, optimizer in enumerate(test_data["Optimizer"].unique()):
            optimizer_data = test_data[test_data["Optimizer"] == optimizer]
            fig.add_trace(
                go.Bar(
                    name=optimizer,
                    x=optimizer_data["Config"],
                    y=optimizer_data["Test Score"],
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    # Update layout
    fig.update_xaxes(title_text="Configuration", row=1, col=1)
    fig.update_xaxes(title_text="Configuration", row=1, col=2)
    fig.update_xaxes(title_text="Total Time (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Configuration", row=2, col=2)

    fig.update_yaxes(title_text="Validation Score", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Validation Score", row=2, col=1)
    fig.update_yaxes(title_text="Test Score", row=2, col=2)

    fig.update_layout(
        title_text="DSPy Optimization Benchmark Results",
        height=800,
        showlegend=True,
        legend=dict(x=1.05, y=1),
    )

    # Save or show
    if output_path:
        fig.write_html(str(output_path))
        print(f"✅ Saved plot to {output_path}")
    else:
        fig.show()


def create_detailed_report(filepath: Path, output_path: Optional[Path] = None) -> str:
    """
    Create a detailed markdown report from benchmark results.

    Args:
        filepath: Path to benchmark results JSON
        output_path: Optional path to save markdown report

    Returns:
        Markdown formatted report string
    """
    data = load_benchmark_results(filepath)

    # Build markdown report
    lines = []
    lines.append("# DSPy Optimization Benchmark Report\n")
    lines.append(f"**Generated:** {data.get('generated_at', 'N/A')}\n")
    lines.append(f"**Number of Benchmarks:** {len(data['benchmarks'])}\n")

    # Summary section
    lines.append("\n## Summary Statistics\n")
    summary = data.get("summary", {})

    for optimizer, stats in summary.get("optimizers", {}).items():
        lines.append(f"\n### {optimizer}\n")
        lines.append(f"- **Runs:** {stats['num_runs']}")
        lines.append(f"- **Average Train Score:** {stats['avg_train_score']:.3f}")
        lines.append(f"- **Average Val Score:** {stats['avg_val_score']:.3f}")
        if "avg_test_score" in stats:
            lines.append(f"- **Average Test Score:** {stats['avg_test_score']:.3f}")
        lines.append(f"- **Average Time:** {stats['avg_time']:.1f}s")
        lines.append(f"- **Best Val Score:** {stats['best_val_score']:.3f}")
        lines.append(f"- **Worst Val Score:** {stats['worst_val_score']:.3f}")

    # Detailed results
    lines.append("\n## Detailed Results\n")
    lines.append("| Optimizer | Config | Train | Val | Test | Time (s) | Examples | Demos |")
    lines.append("|-----------|--------|-------|-----|------|----------|----------|-------|")

    for benchmark in data["benchmarks"]:
        metrics = benchmark["metrics"]
        test_score = (
            f"{metrics['test_score']:.3f}" if metrics.get("test_score") is not None else "N/A"
        )

        lines.append(
            f"| {metrics.get('optimizer_name', 'N/A')} "
            f"| {metrics.get('config_name', 'N/A')} "
            f"| {metrics['train_score']:.3f} "
            f"| {metrics['val_score']:.3f} "
            f"| {test_score} "
            f"| {metrics['total_time']:.1f} "
            f"| {metrics.get('num_examples_used', 0)} "
            f"| {metrics.get('num_demonstrations', 0)} |"
        )

    # Errors section
    lines.append("\n## Errors and Issues\n")
    has_errors = False
    for i, benchmark in enumerate(data["benchmarks"]):
        if benchmark.get("errors"):
            has_errors = True
            config_name = benchmark["metrics"].get("config_name", f"Benchmark {i+1}")
            lines.append(f"\n### {config_name}\n")
            for error in benchmark["errors"]:
                lines.append(f"- {error}")

    if not has_errors:
        lines.append("No errors reported.")

    # Recommendations
    lines.append("\n## Recommendations\n")

    # Find best configuration
    best = max(data["benchmarks"], key=lambda b: b["metrics"]["val_score"])
    lines.append(f"\n### Best Configuration (by validation score)\n")
    lines.append(f"- **Config:** {best['metrics'].get('config_name', 'N/A')}")
    lines.append(f"- **Optimizer:** {best['metrics'].get('optimizer_name', 'N/A')}")
    lines.append(f"- **Val Score:** {best['metrics']['val_score']:.3f}")
    if best["metrics"].get("test_score"):
        lines.append(f"- **Test Score:** {best['metrics']['test_score']:.3f}")
    lines.append(f"- **Time:** {best['metrics']['total_time']:.1f}s")

    # Most efficient
    efficient = max(
        data["benchmarks"],
        key=lambda b: b["metrics"]["val_score"] / max(b["metrics"]["total_time"], 1),
    )
    lines.append(f"\n### Most Efficient Configuration (score/time)\n")
    lines.append(f"- **Config:** {efficient['metrics'].get('config_name', 'N/A')}")
    lines.append(f"- **Optimizer:** {efficient['metrics'].get('optimizer_name', 'N/A')}")
    lines.append(f"- **Val Score:** {efficient['metrics']['val_score']:.3f}")
    lines.append(f"- **Time:** {efficient['metrics']['total_time']:.1f}s")
    efficiency_score = efficient["metrics"]["val_score"] / max(
        efficient["metrics"]["total_time"], 1
    )
    lines.append(f"- **Efficiency:** {efficiency_score:.4f}")

    report = "\n".join(lines)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✅ Saved report to {output_path}")

    return report


def print_quick_summary(filepath: Path):
    """Print a quick summary of benchmark results to console"""
    data = load_benchmark_results(filepath)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\nGenerated: {data.get('generated_at', 'N/A')}")
    print(f"Total benchmarks: {len(data['benchmarks'])}\n")

    # Create simple table
    print(f"{'Optimizer':<20} {'Config':<15} {'Val Score':<12} {'Time (s)':<10}")
    print("-" * 80)

    for benchmark in data["benchmarks"]:
        metrics = benchmark["metrics"]
        print(
            f"{metrics.get('optimizer_name', 'N/A'):<20} "
            f"{metrics.get('config_name', 'N/A'):<15} "
            f"{metrics['val_score']:<12.3f} "
            f"{metrics['total_time']:<10.1f}"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    """Example usage of analysis tools"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark_analysis.py <benchmark_results.json>")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Print quick summary
    print_quick_summary(filepath)

    # Create detailed report
    report_path = filepath.parent / f"{filepath.stem}_report.md"
    create_detailed_report(filepath, report_path)

    # Create plot if plotly available
    if PLOTLY_AVAILABLE:
        plot_path = filepath.parent / f"{filepath.stem}_plot.html"
        df = compare_benchmarks([filepath])
        create_performance_plot(df, plot_path)
