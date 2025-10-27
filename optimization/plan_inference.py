"""
Auto-diff plan inference for Option C dataset creation.

This module provides tools to automatically infer cleaning plans by comparing
raw and cleaned datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import dspy
import pandas as pd

from file_io_tool import FileIOTool
from signatures import CleaningPlanInferenceSignature


class PlanInferenceEngine:
    """
    Infers data cleaning operations by comparing raw and cleaned datasets.

    This is a simple LLM-based approach that analyzes the differences between
    raw and cleaned data to generate a cleaning plan.
    """

    def __init__(self, io_tool: Optional[FileIOTool] = None):
        """
        Initialize the plan inference engine.

        Args:
            io_tool: FileIOTool instance for loading data (creates new if None)
        """
        self.io_tool = io_tool or FileIOTool()
        self.predictor = dspy.ChainOfThought(CleaningPlanInferenceSignature)

    def _summarize_dataframe(
        self, df: pd.DataFrame, label: str = "Dataset", include_issues: bool = True
    ) -> str:
        """
        Create a text summary of a DataFrame for LLM analysis.

        Args:
            df: DataFrame to summarize
            label: Label for this dataset (e.g., "Raw" or "Cleaned")
            include_issues: Whether to include data quality issues

        Returns:
            Text summary of the DataFrame
        """
        summary_lines = []

        # Basic info
        summary_lines.append(f"=== {label} Dataset ===")
        summary_lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        summary_lines.append(f"Columns: {list(df.columns)}")

        # Data types
        summary_lines.append("\nData Types:")
        for col, dtype in df.dtypes.items():
            summary_lines.append(f"  {col}: {dtype}")

        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            summary_lines.append("\nMissing Values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                summary_lines.append(f"  {col}: {count} ({pct:.1f}%)")

        # Sample data (first 3 rows)
        summary_lines.append("\nSample Rows (first 3):")
        sample = df.head(3).to_dict(orient="records")
        for i, row in enumerate(sample, 1):
            summary_lines.append(f"  Row {i}: {row}")

        # Data quality issues (only for raw data)
        if include_issues:
            # Check for duplicates
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                summary_lines.append(f"\nDuplicates: {n_duplicates} duplicate rows found")

            # Check for potential outliers in numeric columns
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
            if len(numeric_cols) > 0:
                summary_lines.append("\nNumeric Column Statistics:")
                for col in numeric_cols:
                    stats = df[col].describe()
                    summary_lines.append(
                        f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                        f"min={stats['min']:.2f}, max={stats['max']:.2f}"
                    )

            # Check for text columns that might need cleaning
            text_cols = df.select_dtypes(include=["object"]).columns
            if len(text_cols) > 0:
                summary_lines.append("\nText Columns (sample unique values):")
                for col in text_cols:
                    unique_sample = df[col].dropna().unique()[:3]
                    summary_lines.append(f"  {col}: {list(unique_sample)}")

        return "\n".join(summary_lines)

    def infer_cleaning_plan(self, raw_path: str, cleaned_path: str, verbose: bool = True) -> str:
        """
        Infer the cleaning plan by comparing raw and cleaned datasets.

        Args:
            raw_path: Path to raw (input) dataset
            cleaned_path: Path to cleaned (output) dataset
            verbose: Print progress information

        Returns:
            Inferred cleaning plan as a string

        Raises:
            FileNotFoundError: If either file doesn't exist
            ValueError: If files can't be loaded or compared
        """
        if verbose:
            print(f"Loading raw data from: {raw_path}")

        # Load raw data
        raw_df = self.io_tool.load_data(raw_path)
        if raw_df is None:
            raise FileNotFoundError(f"Could not load raw data from: {raw_path}")

        if verbose:
            print(f"  Loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
            print(f"Loading cleaned data from: {cleaned_path}")

        # Load cleaned data
        cleaned_df = self.io_tool.load_data(cleaned_path)
        if cleaned_df is None:
            raise FileNotFoundError(f"Could not load cleaned data from: {cleaned_path}")

        if verbose:
            print(f"  Loaded: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
            print("Analyzing differences...")

        # Create summaries
        raw_summary = self._summarize_dataframe(raw_df, "Raw", include_issues=True)
        cleaned_summary = self._summarize_dataframe(cleaned_df, "Cleaned", include_issues=False)

        if verbose:
            print("Inferring cleaning plan with LLM...")

        # Call LLM to infer plan
        try:
            result = self.predictor(
                raw_data_summary=raw_summary, cleaned_data_summary=cleaned_summary
            )
            cleaning_plan = result.cleaning_plan

            if verbose:
                print("✅ Plan inference complete")

            return cleaning_plan

        except Exception as e:
            raise ValueError(f"Failed to infer cleaning plan: {e}")

    def infer_and_format_plan(
        self, raw_path: str, cleaned_path: str, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Infer cleaning plan and return in structured format for CleaningExample.

        Args:
            raw_path: Path to raw dataset
            cleaned_path: Path to cleaned dataset
            verbose: Print progress information

        Returns:
            Dictionary with 'expected_cleaning_plan' key containing the inferred plan
        """
        plan = self.infer_cleaning_plan(raw_path, cleaned_path, verbose=verbose)

        return {"expected_cleaning_plan": plan, "expected_rationale": None}


def infer_plan_from_files(raw_path: str, cleaned_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to infer cleaning plan from file paths.

    Args:
        raw_path: Path to raw dataset
        cleaned_path: Path to cleaned dataset
        verbose: Print progress information

    Returns:
        Dictionary with inferred plan suitable for CleaningExample

    Example:
        >>> result = infer_plan_from_files("data/raw.csv", "data/cleaned.csv")
        >>> example = CleaningExample(
        ...     input_path="data/raw.csv",
        ...     **result
        ... )
    """
    engine = PlanInferenceEngine()
    return engine.infer_and_format_plan(raw_path, cleaned_path, verbose=verbose)


if __name__ == "__main__":
    """Example usage of plan inference"""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python plan_inference.py <raw_file> <cleaned_file>")
        print("\nExample:")
        print("  python plan_inference.py data/raw.csv data/cleaned.csv")
        sys.exit(1)

    raw_file = sys.argv[1]
    cleaned_file = sys.argv[2]

    # Configure DSPy (would need LLM config in real usage)
    print("Note: Make sure DSPy is configured with an LLM before running this.")
    print("Example: dspy.settings.configure(lm=your_lm)\n")

    # Infer plan
    try:
        result = infer_plan_from_files(raw_file, cleaned_file, verbose=True)
        print("\n" + "=" * 80)
        print("INFERRED CLEANING PLAN")
        print("=" * 80)
        print(result["expected_cleaning_plan"])
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
