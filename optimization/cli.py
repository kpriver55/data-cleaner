"""
Command-Line Interface for DSPy Optimization

Provides commands for:
- Creating and managing training datasets
- Validating datasets
- Running optimizations
- Evaluating models
- Comparing models

Usage:
    python -m optimization.cli <command> [options]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import time

# Rich library for beautiful terminal output (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_header(text: str):
    """Print a header"""
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]{text}[/bold cyan]")
    else:
        print(f"\n{'='*60}\n{text}\n{'='*60}")


def print_success(text: str):
    """Print success message"""
    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] {text}")
    else:
        print(f"✓ {text}")


def print_error(text: str):
    """Print error message"""
    if RICH_AVAILABLE:
        console.print(f"[red]✗[/red] {text}")
    else:
        print(f"✗ {text}")


def print_info(text: str):
    """Print info message"""
    if RICH_AVAILABLE:
        console.print(f"[blue]ℹ[/blue] {text}")
    else:
        print(f"ℹ {text}")


def cmd_create_dataset(args):
    """Create a new training dataset"""
    from optimization import CleaningDataset

    print_header("Create New Training Dataset")

    output_path = Path(args.output)

    # Check if file exists
    if output_path.exists() and not args.force:
        print_error(f"File already exists: {output_path}")
        print_info("Use --force to overwrite")
        return 1

    # Create empty dataset
    dataset = CleaningDataset()

    # Save it
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'json' or output_path.suffix == '.json':
        dataset.to_json(output_path)
    else:
        dataset.to_yaml(output_path)

    print_success(f"Created empty dataset: {output_path}")
    print_info("Add examples with: python -m optimization.cli add-example")

    return 0


def cmd_add_example(args):
    """Add an example to a dataset"""
    from optimization import CleaningDataset, CleaningExample

    print_header("Add Example to Dataset")

    dataset_path = Path(args.dataset)

    # Load existing dataset
    if not dataset_path.exists():
        print_error(f"Dataset not found: {dataset_path}")
        return 1

    if dataset_path.suffix == '.json':
        dataset = CleaningDataset.from_json(dataset_path)
    else:
        dataset = CleaningDataset.from_yaml(dataset_path)

    print_info(f"Loaded dataset with {len(dataset)} examples")

    # Create example
    example_data = {
        'input_path': args.input
    }

    # Add optional fields
    if args.plan:
        with open(args.plan, 'r') as f:
            example_data['expected_cleaning_plan'] = f.read()

    if args.rationale:
        with open(args.rationale, 'r') as f:
            example_data['expected_rationale'] = f.read()

    if args.operations:
        with open(args.operations, 'r') as f:
            example_data['expected_operations'] = json.load(f)

    if args.description:
        example_data['description'] = args.description

    # Create and add example
    try:
        example = CleaningExample(**example_data)
        dataset.add_example(example)
        print_success(f"Added example: {args.input}")
    except Exception as e:
        print_error(f"Failed to create example: {e}")
        return 1

    # Save dataset
    if dataset_path.suffix == '.json':
        dataset.to_json(dataset_path)
    else:
        dataset.to_yaml(dataset_path)

    print_success(f"Saved dataset with {len(dataset)} examples")

    return 0


def cmd_validate(args):
    """Validate a training dataset"""
    from optimization import CleaningDataset

    print_header("Validate Training Dataset")

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print_error(f"Dataset not found: {dataset_path}")
        return 1

    # Load dataset
    if dataset_path.suffix == '.json':
        dataset = CleaningDataset.from_json(dataset_path)
    else:
        dataset = CleaningDataset.from_yaml(dataset_path)

    print_info(f"Loaded {len(dataset)} examples")

    # Validate
    errors = dataset.validate()

    if errors:
        print_error(f"Validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"  • {error}")
        return 1
    else:
        print_success("Dataset is valid!")

        # Show summary
        summary = dataset.summary()
        if RICH_AVAILABLE:
            table = Table(title="Dataset Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total examples", str(summary['total_examples']))
            table.add_row("Full plan specs", str(summary['specification_types']['plan']))
            table.add_row("Structured ops", str(summary['specification_types']['operations']))
            table.add_row("With descriptions", str(summary['examples_with_description']))

            console.print(table)
        else:
            print("\nDataset Summary:")
            print(f"  Total examples: {summary['total_examples']}")
            print(f"  Full plan specs: {summary['specification_types']['plan']}")
            print(f"  Structured ops: {summary['specification_types']['operations']}")
            print(f"  With descriptions: {summary['examples_with_description']}")

        return 0


def cmd_optimize(args):
    """Run optimization"""
    from optimization import OptimizationConfig
    from data_cleaning_agent import DataCleaningAgent
    from file_io_tool import FileIOTool
    from stats_tool import StatisticalAnalysisTool
    from data_transformation_tool import DataTransformationTool
    from llm_config import setup_llm

    print_header("Run Optimization")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print_error(f"Configuration not found: {config_path}")
        return 1

    if config_path.suffix == '.json':
        config = OptimizationConfig.load_json(config_path)
    else:
        config = OptimizationConfig.load_yaml(config_path)

    print_info(f"Loaded configuration: {config.name}")
    print_info(f"Optimizer: {config.optimizer.optimizer_type}")
    print_info(f"Dataset: {config.dataset_path}")

    # Setup LLM if specified
    if args.llm_provider:
        print_info(f"Configuring LLM: {args.llm_provider}/{args.llm_model or 'default'}")
        try:
            setup_llm(args.llm_provider, args.llm_model)
        except Exception as e:
            print_error(f"Failed to configure LLM: {e}")
            return 1

    # Create agent
    print_info("Creating DataCleaningAgent...")
    io_tool = FileIOTool()
    stats_tool = StatisticalAnalysisTool()
    transform_tool = DataTransformationTool()
    agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)

    # Run optimization
    print_info("Starting optimization (this may take several minutes)...")

    start_time = time.time()

    try:
        result = agent.compile_with_optimizer(config, verbose=not args.quiet)
        duration = time.time() - start_time

        print_success(f"Optimization completed in {duration:.2f} seconds")
        print_info(f"Training score: {result.train_score:.4f}")
        if result.val_score is not None:
            print_info(f"Validation score: {result.val_score:.4f}")

    except Exception as e:
        print_error(f"Optimization failed: {e}")
        return 1

    # Save model
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        agent.save_compiled_model(
            str(output_path),
            metadata={
                'config': config.to_dict(),
                'result': result.to_dict()
            }
        )
        print_success(f"Model saved to: {output_path}")

    return 0


def cmd_evaluate(args):
    """Evaluate a model on a test dataset"""
    from optimization import CleaningDataset
    from data_cleaning_agent import DataCleaningAgent
    from file_io_tool import FileIOTool
    from stats_tool import StatisticalAnalysisTool
    from data_transformation_tool import DataTransformationTool

    print_header("Evaluate Model")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print_error(f"Model not found: {model_path}")
        return 1

    print_info(f"Loading model from: {model_path}")

    io_tool = FileIOTool()
    stats_tool = StatisticalAnalysisTool()
    transform_tool = DataTransformationTool()

    agent = DataCleaningAgent.load_compiled_model(
        str(model_path),
        io_tool,
        stats_tool,
        transform_tool
    )

    # Load test dataset
    test_path = Path(args.test_dataset)
    if not test_path.exists():
        print_error(f"Test dataset not found: {test_path}")
        return 1

    if test_path.suffix == '.json':
        dataset = CleaningDataset.from_json(test_path)
    else:
        dataset = CleaningDataset.from_yaml(test_path)

    print_info(f"Loaded {len(dataset)} test examples")

    # Evaluate
    print_info("Evaluating...")

    results = agent.evaluate_on_dataset(dataset, verbose=not args.quiet)

    # Display results
    print_success("Evaluation complete")

    if RICH_AVAILABLE:
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Average Score", f"{results['average_score']:.4f}")
        table.add_row("Min Score", f"{results['min_score']:.4f}")
        table.add_row("Max Score", f"{results['max_score']:.4f}")
        table.add_row("Examples", str(results['num_examples']))

        console.print(table)
    else:
        print("\nEvaluation Results:")
        print(f"  Average Score: {results['average_score']:.4f}")
        print(f"  Min Score: {results['min_score']:.4f}")
        print(f"  Max Score: {results['max_score']:.4f}")
        print(f"  Examples: {results['num_examples']}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print_success(f"Results saved to: {output_path}")

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DSPy Optimization CLI for Data Cleaning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # create-dataset command
    create_parser = subparsers.add_parser(
        'create-dataset',
        help='Create a new training dataset'
    )
    create_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output path for the dataset (JSON or YAML)'
    )
    create_parser.add_argument(
        '--format',
        choices=['json', 'yaml'],
        default='json',
        help='Output format (default: json)'
    )
    create_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing file'
    )

    # add-example command
    add_parser = subparsers.add_parser(
        'add-example',
        help='Add an example to a dataset'
    )
    add_parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Path to the dataset file'
    )
    add_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the messy input CSV/Excel file'
    )
    add_parser.add_argument(
        '--plan',
        help='Path to file containing expected cleaning plan (Option A)'
    )
    add_parser.add_argument(
        '--rationale',
        help='Path to file containing expected rationale (Option A)'
    )
    add_parser.add_argument(
        '--operations',
        help='Path to JSON file with expected operations (Option B)'
    )
    add_parser.add_argument(
        '--description',
        help='Description of the example'
    )

    # validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a training dataset'
    )
    validate_parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Path to the dataset file'
    )

    # optimize command
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Run optimization'
    )
    optimize_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to optimization configuration (JSON or YAML)'
    )
    optimize_parser.add_argument(
        '--output', '-o',
        help='Output path for optimized model (.pkl)'
    )
    optimize_parser.add_argument(
        '--llm-provider',
        help='LLM provider (ollama, openai, anthropic, etc.)'
    )
    optimize_parser.add_argument(
        '--llm-model',
        help='LLM model name'
    )
    optimize_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    # evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a model on a test dataset'
    )
    evaluate_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to the model file (.pkl)'
    )
    evaluate_parser.add_argument(
        '--test-dataset', '-t',
        required=True,
        help='Path to test dataset (JSON or YAML)'
    )
    evaluate_parser.add_argument(
        '--output', '-o',
        help='Output path for results (JSON)'
    )
    evaluate_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handler
    commands = {
        'create-dataset': cmd_create_dataset,
        'add-example': cmd_add_example,
        'validate': cmd_validate,
        'optimize': cmd_optimize,
        'evaluate': cmd_evaluate
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print_error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
