import dspy
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path






class DataCleaningTool:
    """Wrapper class to make our transformation tools compatible with ReAct"""
    
    def __init__(self, transform_tool, stats_tool):
        self.transform_tool = transform_tool
        self.stats_tool = stats_tool
        self.current_df = None
        self.operations_log = []
    
    def set_current_dataframe(self, df: pd.DataFrame):
        """Set the dataframe that operations will work on"""
        self.current_df = df.copy()
    
    def get_current_dataframe(self) -> pd.DataFrame:
        """Get the current state of the dataframe"""
        return self.current_df.copy()
    
    def handle_missing_values(self,
                            numeric_strategy: str = 'median',
                            categorical_strategy: str = 'most_frequent',
                            columns: Optional[List[str]] = None) -> str:
        """Handle missing values in the dataset with separate strategies for numeric and categorical columns"""
        try:
            if self.current_df is None:
                return "Error: No dataframe loaded"
            
            original_rows = len(self.current_df)

            self.current_df = self.transform_tool.handle_missing_values(
                self.current_df,
                strategy=None,  # Always use separate strategies
                numeric_strategy=numeric_strategy,
                categorical_strategy=categorical_strategy,
                columns=columns
            )

            result = f"Successfully handled missing values using '{numeric_strategy}' for numeric and '{categorical_strategy}' for categorical columns. "

            if columns:
                result += f"Applied to columns: {columns}. "
            else:
                result += "Applied to all columns with missing values. "
            result += f"Dataset still has {len(self.current_df)} rows."

            self.operations_log.append({
                'operation': 'handle_missing_values',
                'parameters': {
                    'numeric_strategy': numeric_strategy,
                    'categorical_strategy': categorical_strategy,
                    'columns': columns
                },
                'rows_before': original_rows,
                'rows_after': len(self.current_df)
            })
            
            return result
            
        except Exception as e:
            return f"Error handling missing values: {str(e)}"
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> str:
        """Remove duplicate rows from the dataset"""
        try:
            if self.current_df is None:
                return "Error: No dataframe loaded"
            
            original_rows = len(self.current_df)
            self.current_df = self.transform_tool.remove_duplicates(
                self.current_df, subset=subset, keep=keep
            )
            
            duplicates_removed = original_rows - len(self.current_df)
            result = f"Removed {duplicates_removed} duplicate rows. Dataset now has {len(self.current_df)} rows."
            
            self.operations_log.append({
                'operation': 'remove_duplicates',
                'parameters': {'subset': subset, 'keep': keep},
                'rows_before': original_rows,
                'rows_after': len(self.current_df)
            })
            
            return result
            
        except Exception as e:
            return f"Error removing duplicates: {str(e)}"
    
    def remove_outliers(self, columns: Optional[List[str]] = None, method: str = 'iqr', threshold: float = 1.5) -> str:
        """Remove outliers from numeric columns"""
        try:
            if self.current_df is None:
                return "Error: No dataframe loaded"
            
            original_rows = len(self.current_df)
            self.current_df, outlier_report = self.transform_tool.remove_outliers(
                self.current_df, columns=columns, method=method, threshold=threshold
            )
            
            outliers_removed = original_rows - len(self.current_df)
            result = f"Removed {outliers_removed} outlier rows using {method} method with threshold {threshold}. "
            result += f"Dataset now has {len(self.current_df)} rows. "
            result += f"Outlier report: {outlier_report}"
            
            self.operations_log.append({
                'operation': 'remove_outliers',
                'parameters': {'columns': columns, 'method': method, 'threshold': threshold},
                'rows_before': original_rows,
                'rows_after': len(self.current_df)
            })
            
            return result
            
        except Exception as e:
            return f"Error removing outliers: {str(e)}"
    
    def clean_text_columns(self, columns: Optional[List[str]] = None, operations: List[str] = None) -> str:
        """Clean text columns"""
        try:
            if self.current_df is None:
                return "Error: No dataframe loaded"
            
            if operations is None:
                operations = ['strip', 'lower']
            
            original_rows = len(self.current_df)
            self.current_df = self.transform_tool.clean_text_columns(
                self.current_df, columns=columns, operations=operations
            )
            
            result = f"Cleaned text columns using operations: {operations}. "
            if columns:
                result += f"Applied to columns: {columns}. "
            else:
                result += "Applied to all text columns. "
            result += f"Dataset still has {len(self.current_df)} rows."
            
            self.operations_log.append({
                'operation': 'clean_text_columns',
                'parameters': {'columns': columns, 'operations': operations},
                'rows_before': original_rows,
                'rows_after': len(self.current_df)
            })
            
            return result
            
        except Exception as e:
            return f"Error cleaning text columns: {str(e)}"
    
    def convert_data_types(self, type_conversions: Dict[str, str]) -> str:
        """Convert data types of columns"""
        try:
            if self.current_df is None:
                return "Error: No dataframe loaded"
            
            original_rows = len(self.current_df)
            self.current_df = self.transform_tool.convert_data_types(
                self.current_df, type_conversions
            )
            
            result = f"Converted data types: {type_conversions}. Dataset still has {len(self.current_df)} rows."
            
            self.operations_log.append({
                'operation': 'convert_data_types',
                'parameters': {'type_conversions': type_conversions},
                'rows_before': original_rows,
                'rows_after': len(self.current_df)
            })
            
            return result
            
        except Exception as e:
            return f"Error converting data types: {str(e)}"
    
    def inspect_data(self, columns: Optional[List[str]] = None) -> str:
        """Inspect current state of the data"""
        try:
            if self.current_df is None:
                return "Error: No dataframe loaded"
            
            result = f"Dataset shape: {self.current_df.shape[0]} rows × {self.current_df.shape[1]} columns\n"
            
            if columns:
                inspect_cols = [col for col in columns if col in self.current_df.columns]
            else:
                inspect_cols = self.current_df.columns[:5]  # First 5 columns
            
            for col in inspect_cols:
                missing_count = self.current_df[col].isna().sum()
                missing_pct = (missing_count / len(self.current_df)) * 100
                result += f"Column '{col}': {self.current_df[col].dtype}, {missing_count} missing ({missing_pct:.1f}%)\n"
                
                if pd.api.types.is_numeric_dtype(self.current_df[col]):
                    outlier_info = self.stats_tool.detect_outliers(self.current_df, col)
                    result += f"  Outliers: {outlier_info.get('outlier_count', 0)}\n"
            
            return result
            
        except Exception as e:
            return f"Error inspecting data: {str(e)}"
    
    def get_operations_log(self) -> str:
        """Get log of all operations performed"""
        if not self.operations_log:
            return "No operations performed yet."

        result = "Operations performed:\n"
        for i, op in enumerate(self.operations_log, 1):
            result += f"{i}. {op['operation']}: {op['rows_before']} → {op['rows_after']} rows\n"

        return result

    def validate_completion(self, required_operations: List[str]) -> str:
        """Validate that all required operations have been performed"""
        try:
            if not required_operations:
                return "No required operations specified."

            performed_ops = [op['operation'] for op in self.operations_log]
            missing_ops = []

            for required_op in required_operations:
                if required_op not in performed_ops:
                    missing_ops.append(required_op)

            if missing_ops:
                return f"VALIDATION FAILED: You have NOT completed all required operations. Missing operations: {missing_ops}. You MUST execute these operations before claiming completion. Use the appropriate tools to perform these missing operations."
            else:
                return f"VALIDATION PASSED: All required operations have been performed successfully. Operations completed: {performed_ops}. You may now finish."

        except Exception as e:
            return f"Error during validation: {str(e)}"