"""
Dataset Management for DSPy Optimization

Manages training and test examples for optimizing the data cleaning agent.
Supports loading examples from various formats and creating DSPy Example objects.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import dspy


@dataclass
class CleaningExample:
    """
    Represents a single training/test example for data cleaning optimization

    Supports three levels of specification (in order of preference):
    A. Full plan specification (best quality - use for most important examples)
    B. Structured operations (good quality - balance of effort and quality)
    C. Input/output comparison (TODO - not yet implemented)

    Attributes:
        input_path: Path to the messy input dataset

        # Option A: Provide complete expected plan (highest quality)
        expected_cleaning_plan: Full text of the expected cleaning plan
        expected_rationale: Full text explaining the rationale

        # Option B: Provide structured operations (good quality)
        expected_operations: List of dicts with operation details including:
            - operation: name of the operation
            - columns: which columns to apply to
            - parameters: strategy/method/threshold etc.
            - rationale: why this operation is needed

        # Option C: Provide expected output for comparison (TODO)
        expected_output_path: Path to the expected cleaned dataset

        metadata: Additional metadata about the example
        description: Human-readable description of the cleaning task
    """
    input_path: str

    # Option A: Full plan specification (best)
    expected_cleaning_plan: Optional[str] = None
    expected_rationale: Optional[str] = None

    # Option B: Structured operations (good)
    expected_operations: Optional[List[Dict[str, Any]]] = None

    # Option C: Output comparison (TODO - not yet implemented)
    expected_output_path: Optional[str] = None

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate that at least one expected output is provided"""
        if not any([
            self.expected_cleaning_plan,
            self.expected_operations,
            self.expected_output_path
        ]):
            raise ValueError(
                "Must provide at least one of: expected_cleaning_plan, "
                "expected_operations, or expected_output_path"
            )

        # Validate that if plan is provided, rationale is also provided
        if self.expected_cleaning_plan and not self.expected_rationale:
            raise ValueError(
                "If expected_cleaning_plan is provided, expected_rationale must also be provided"
            )

        # Validate structured operations format if provided
        if self.expected_operations:
            for i, op in enumerate(self.expected_operations):
                if 'operation' not in op:
                    raise ValueError(f"Operation {i} missing required 'operation' field")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'input_path': self.input_path,
            'expected_cleaning_plan': self.expected_cleaning_plan,
            'expected_rationale': self.expected_rationale,
            'expected_operations': self.expected_operations,
            'expected_output_path': self.expected_output_path,
            'metadata': self.metadata,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CleaningExample':
        """Create from dictionary"""
        return cls(**data)

    def get_specification_type(self) -> str:
        """
        Determine which type of specification this example uses

        Returns:
            'plan' (Option A), 'operations' (Option B), or 'output' (Option C)
        """
        if self.expected_cleaning_plan and self.expected_rationale:
            return 'plan'
        elif self.expected_operations:
            return 'operations'
        elif self.expected_output_path:
            return 'output'
        else:
            return 'unknown'


class CleaningDataset:
    """
    Manages a collection of cleaning examples for training and evaluation
    """

    def __init__(self, examples: Optional[List[CleaningExample]] = None):
        """
        Initialize dataset

        Args:
            examples: Optional list of CleaningExample objects
        """
        self.examples = examples or []

    def add_example(self, example: CleaningExample):
        """Add a single example to the dataset"""
        self.examples.append(example)

    def __len__(self) -> int:
        """Get number of examples in dataset"""
        return len(self.examples)

    def __getitem__(self, idx: int) -> CleaningExample:
        """Get example by index"""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples"""
        return iter(self.examples)

    def split(self, train_ratio: float = 0.8) -> tuple['CleaningDataset', 'CleaningDataset']:
        """
        Split dataset into train and test sets

        Args:
            train_ratio: Ratio of examples to use for training (default: 0.8)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        split_idx = int(len(self.examples) * train_ratio)
        train_examples = self.examples[:split_idx]
        test_examples = self.examples[split_idx:]

        return CleaningDataset(train_examples), CleaningDataset(test_examples)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'CleaningDataset':
        """
        Load dataset from JSON file

        JSON format examples:

        Option A (Full plan specification - highest quality):
        {
            "examples": [
                {
                    "input_path": "data/messy.csv",
                    "expected_cleaning_plan": "1. remove_duplicates(subset=['id', 'email'], keep='first')\\n2. handle_missing_values(columns=['age', 'salary'], numeric_strategy='median')\\n3. clean_text_columns(columns=['email'], operations=['strip', 'lower'])",
                    "expected_rationale": "First remove duplicates based on ID and email. Then impute missing numeric values with median. Finally normalize email addresses.",
                    "description": "Clean customer data"
                }
            ]
        }

        Option B (Structured operations - good quality):
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
                            "rationale": "Impute with median to preserve distribution"
                        }
                    ],
                    "description": "Clean customer data"
                }
            ]
        }

        Args:
            json_path: Path to JSON file

        Returns:
            CleaningDataset object
        """
        json_path = Path(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)

        examples = [CleaningExample.from_dict(ex) for ex in data.get('examples', [])]
        return cls(examples)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'CleaningDataset':
        """
        Load dataset from YAML file

        YAML format similar to JSON format (see from_json docstring)

        Args:
            yaml_path: Path to YAML file

        Returns:
            CleaningDataset object
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        examples = [CleaningExample.from_dict(ex) for ex in data.get('examples', [])]
        return cls(examples)

    def to_json(self, json_path: Union[str, Path]):
        """
        Save dataset to JSON file

        Args:
            json_path: Path to save JSON file
        """
        json_path = Path(json_path)
        data = {
            'examples': [ex.to_dict() for ex in self.examples]
        }

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save dataset to YAML file

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        data = {
            'examples': [ex.to_dict() for ex in self.examples]
        }

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dspy_examples(self, io_tool) -> List[dspy.Example]:
        """
        Convert cleaning examples to DSPy Example objects for optimization

        This prepares examples in the format expected by DSPy optimizers.
        For the planning stage, we create examples with data analysis input
        and expected cleaning plan output.

        Args:
            io_tool: FileIOTool instance for reading datasets

        Returns:
            List of dspy.Example objects
        """
        from stats_tool import StatisticalAnalysisTool

        stats_tool = StatisticalAnalysisTool()
        dspy_examples = []

        for i, example in enumerate(self.examples):
            # Load the input dataset
            try:
                df = io_tool.read_spreadsheet(example.input_path)
            except Exception as e:
                print(f"Warning: Could not load {example.input_path}: {e}")
                continue

            # Generate comprehensive data analysis
            comprehensive_analysis = self._generate_data_analysis(df, stats_tool)

            # Generate expected plan based on what the user provided
            try:
                expected_plan = self._generate_expected_plan(example, df)
            except NotImplementedError as e:
                print(f"Warning: Skipping example {i}: {e}")
                continue

            # Create DSPy Example with inputs marked
            dspy_example = dspy.Example(
                comprehensive_data_analysis=comprehensive_analysis,
                overall_cleaning_plan=expected_plan['overall_cleaning_plan'],
                rationale=expected_plan['rationale']
            ).with_inputs('comprehensive_data_analysis')

            dspy_examples.append(dspy_example)

        return dspy_examples

    def _generate_data_analysis(self, df: pd.DataFrame, stats_tool) -> str:
        """
        Generate comprehensive data analysis string for a dataframe

        Args:
            df: Input dataframe
            stats_tool: StatisticalAnalysisTool instance

        Returns:
            Comprehensive analysis string
        """
        # Get comprehensive data summary
        summary_report = stats_tool.summary_report(df)

        # Get missing data details
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / len(df)) * 100, 2),
                    'dtype': str(df[col].dtype)
                }

        # Get data type information
        dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Sample some actual data for context
        sample_data = {}
        for col in df.columns:
            non_null_values = df[col].dropna().head(5).tolist()
            sample_data[col] = [str(val) for val in non_null_values]

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            else:
                return obj

        column_stats = convert_numpy_types(summary_report['column_statistics'])
        sample_data = convert_numpy_types(sample_data)

        # Build comprehensive analysis string
        comprehensive_analysis = f"""
=== DATASET OVERVIEW ===
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Info: {json.dumps(summary_report['dataset_info'], indent=2)}

=== DUPLICATE ANALYSIS ===
{json.dumps(summary_report['duplicate_summary'], indent=2)}

=== MISSING DATA ANALYSIS ===
Column-specific missing data: {json.dumps(missing_info, indent=2)}
Overall missing data summary: {json.dumps(summary_report['missing_data_summary'], indent=2)}

=== DATA TYPES ANALYSIS ===
Current data types: {json.dumps(dtypes_info, indent=2)}

=== COLUMN STATISTICS ===
{json.dumps(column_stats, indent=2)}

=== SAMPLE DATA ===
{json.dumps(sample_data, indent=2)}
"""
        return comprehensive_analysis

    def _generate_expected_plan(self, example: CleaningExample, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate expected cleaning plan from example based on specification type

        Args:
            example: CleaningExample object
            df: Input dataframe (used for validation)

        Returns:
            Dictionary with 'overall_cleaning_plan' and 'rationale' keys
        """
        spec_type = example.get_specification_type()

        # Priority 1: Use human-provided plan (Option A - best quality)
        if spec_type == 'plan':
            return {
                'overall_cleaning_plan': example.expected_cleaning_plan,
                'rationale': example.expected_rationale
            }

        # Priority 2: Generate detailed plan from structured operations (Option B - good quality)
        elif spec_type == 'operations':
            return self._generate_plan_from_structured_ops(example.expected_operations, df)

        # Priority 3: Auto-generate from input/output comparison (Option C - TODO)
        elif spec_type == 'output':
            # TODO: Implement automatic plan generation by comparing input and output datasets
            # This would involve:
            # 1. Loading both input_path and expected_output_path
            # 2. Detecting differences (rows removed, values changed, types converted, etc.)
            # 3. Inferring which operations were performed and their parameters
            # 4. Generating a detailed plan that explains the transformations
            #
            # Challenges:
            # - Determining operation order from final state
            # - Distinguishing between similar operations (e.g., drop vs filter)
            # - Inferring parameters (e.g., which imputation strategy was used)
            # - Handling ambiguous cases where multiple operation sequences could produce same result
            #
            # This would be very valuable as it requires minimal manual specification,
            # but is complex to implement correctly.
            raise NotImplementedError(
                "Automatic plan generation from input/output comparison (Option C) is not yet implemented. "
                "Please use Option A (expected_cleaning_plan + expected_rationale) or "
                "Option B (expected_operations) for now."
            )

        else:
            raise ValueError(f"Unknown specification type: {spec_type}")

    def _generate_plan_from_structured_ops(
        self,
        operations: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Generate a detailed cleaning plan from structured operation specifications

        Args:
            operations: List of operation dictionaries with details
            df: Input dataframe (for validation)

        Returns:
            Dictionary with 'overall_cleaning_plan' and 'rationale'
        """
        plan_lines = []
        rationale_lines = []

        for i, op_spec in enumerate(operations, 1):
            operation = op_spec['operation']

            # Build parameter string based on what's provided
            params = []

            # Extract common parameters
            if 'columns' in op_spec:
                columns = op_spec['columns']
                if columns:  # Only add if not empty
                    params.append(f"columns={columns}")

            # Operation-specific parameters
            if operation == 'handle_missing_values':
                if 'numeric_strategy' in op_spec:
                    params.append(f"numeric_strategy='{op_spec['numeric_strategy']}'")
                if 'categorical_strategy' in op_spec:
                    params.append(f"categorical_strategy='{op_spec['categorical_strategy']}'")

            elif operation == 'remove_duplicates':
                if 'subset' in op_spec:
                    params.append(f"subset={op_spec['subset']}")
                if 'keep' in op_spec:
                    params.append(f"keep='{op_spec['keep']}'")

            elif operation == 'remove_outliers':
                if 'method' in op_spec:
                    params.append(f"method='{op_spec['method']}'")
                if 'threshold' in op_spec:
                    params.append(f"threshold={op_spec['threshold']}")

            elif operation == 'clean_text_columns':
                if 'operations' in op_spec:
                    params.append(f"operations={op_spec['operations']}")

            elif operation == 'convert_data_types':
                if 'type_conversions' in op_spec:
                    params.append(f"type_conversions={op_spec['type_conversions']}")

            # Add any other custom parameters
            for key, value in op_spec.items():
                if key not in ['operation', 'columns', 'rationale', 'numeric_strategy',
                               'categorical_strategy', 'subset', 'keep', 'method',
                               'threshold', 'operations', 'type_conversions']:
                    if isinstance(value, str):
                        params.append(f"{key}='{value}'")
                    else:
                        params.append(f"{key}={value}")

            # Build the operation line
            params_str = ', '.join(params)
            operation_line = f"{i}. {operation}({params_str})"
            plan_lines.append(operation_line)

            # Add rationale if provided
            if 'rationale' in op_spec and op_spec['rationale']:
                rationale_lines.append(f"Step {i}: {op_spec['rationale']}")

        overall_plan = '\n'.join(plan_lines)

        # Build comprehensive rationale
        if rationale_lines:
            rationale = '\n'.join(rationale_lines)
        else:
            rationale = f"Execute {len(operations)} cleaning operations in the specified order to improve data quality."

        return {
            'overall_cleaning_plan': overall_plan,
            'rationale': rationale
        }

    def validate(self) -> List[str]:
        """
        Validate all examples in the dataset

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        for i, example in enumerate(self.examples):
            # Check if input file exists
            if not Path(example.input_path).exists():
                errors.append(f"Example {i}: Input file does not exist: {example.input_path}")

            # Check if expected output file exists (if provided)
            if example.expected_output_path and not Path(example.expected_output_path).exists():
                errors.append(
                    f"Example {i}: Expected output file does not exist: {example.expected_output_path}"
                )

            # Validate specification type
            spec_type = example.get_specification_type()
            if spec_type == 'unknown':
                errors.append(
                    f"Example {i}: Could not determine specification type. "
                    "Provide expected_cleaning_plan, expected_operations, or expected_output_path."
                )

            # Validate Option A (plan specification)
            if example.expected_cleaning_plan and not example.expected_rationale:
                errors.append(
                    f"Example {i}: expected_cleaning_plan provided without expected_rationale"
                )

            # Validate Option B (structured operations)
            if example.expected_operations:
                for j, op in enumerate(example.expected_operations):
                    if 'operation' not in op:
                        errors.append(f"Example {i}, Operation {j}: Missing 'operation' field")

        return errors

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the dataset

        Returns:
            Dictionary with summary information
        """
        spec_types = {
            'plan': 0,
            'operations': 0,
            'output': 0,
            'unknown': 0
        }

        for example in self.examples:
            spec_type = example.get_specification_type()
            spec_types[spec_type] = spec_types.get(spec_type, 0) + 1

        return {
            'total_examples': len(self.examples),
            'specification_types': spec_types,
            'examples_with_description': sum(1 for ex in self.examples if ex.description),
            'examples_with_metadata': sum(1 for ex in self.examples if ex.metadata)
        }
