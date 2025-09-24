import dspy
import pandas as pd
import json
from typing import Dict, Any, Optional
from signatures import DataAnalysisSignature, DataCleaningExecutionSignature
from data_cleaning_tool import DataCleaningTool
from pathlib import Path

class DataCleaningAgent(dspy.Module):
    """
    DSPy agent for data cleaning using ReAct for tool execution
    """
    
    def __init__(self, io_tool, stats_tool, transform_tool):
        super().__init__()
        self.io_tool = io_tool
        self.stats_tool = stats_tool
        self.transform_tool = transform_tool
        
        # Create the cleaning tool wrapper
        self.cleaning_tool = DataCleaningTool(transform_tool, stats_tool)
        
        # DSPy modules
        self.analyzer = dspy.ChainOfThought(DataAnalysisSignature)
        
        # ReAct agent with our cleaning tools
        tools = [
            self.cleaning_tool.handle_missing_values,
            self.cleaning_tool.remove_duplicates,
            self.cleaning_tool.remove_outliers,
            self.cleaning_tool.clean_text_columns,
            self.cleaning_tool.convert_data_types,
            self.cleaning_tool.inspect_data,
            self.cleaning_tool.get_operations_log
        ]
        
        self.react_agent = dspy.ReAct(DataCleaningExecutionSignature, tools=tools)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Use the stats tool to gather information about the dataset"""
        
        # Get comprehensive data summary
        summary_report = self.stats_tool.summary_report(df)
        
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
        
        return {
            'summary_report': summary_report,
            'missing_info': missing_info,
            'dtypes_info': dtypes_info,
            'sample_data': sample_data,
            'shape': df.shape
        }
    
    def get_cleaning_plan(self, data_analysis: Dict[str, Any]) -> tuple:
        """Use LLM reasoning to create a cleaning plan"""
        
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
        
        # Prepare inputs for the analyzer
        column_stats = convert_numpy_types(data_analysis['summary_report']['column_statistics'])
        sample_data = convert_numpy_types(data_analysis['sample_data'])
        
        data_summary = f"""
Dataset Shape: {data_analysis['shape'][0]} rows × {data_analysis['shape'][1]} columns

Dataset Info: {data_analysis['summary_report']['dataset_info']}

Column Statistics: {json.dumps(column_stats, indent=2)}

Sample Data: {json.dumps(sample_data, indent=2)}
"""
        
        missing_data_info = f"""
Missing Data Summary: {json.dumps(data_analysis['missing_info'], indent=2)}

Overall Missing Data: {data_analysis['summary_report']['missing_data_summary']}
"""
        
        data_types_info = f"""
Current Data Types: {json.dumps(data_analysis['dtypes_info'], indent=2)}
"""
        
        # Get LLM's analysis and plan
        result = self.analyzer(
            data_summary=data_summary,
            missing_data_info=missing_data_info,
            data_types_info=data_types_info
        )
        
        return result.cleaning_plan, result.rationale
    
    def clean_dataset(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to clean a dataset using ReAct agent
        """
        print(f"🔍 Loading dataset from: {input_path}")
        
        # Load the data
        df = self.io_tool.read_spreadsheet(input_path)
        original_shape = df.shape
        
        print(f"📊 Original dataset: {original_shape[0]} rows × {original_shape[1]} columns")
        
        # Set the dataframe in our cleaning tool
        self.cleaning_tool.set_current_dataframe(df)
        
        # Analyze data quality
        print("🔍 Analyzing data quality...")
        data_analysis = self.analyze_data_quality(df)
        
        # Get cleaning plan from LLM
        print("🧠 Generating cleaning plan...")
        cleaning_plan, rationale = self.get_cleaning_plan(data_analysis)
        
        print("📋 Cleaning Plan:")
        print(cleaning_plan)
        print(f"\n💭 Rationale:")
        print(rationale)
        
        # Use ReAct agent to execute the cleaning plan
        print(f"\n🚀 Executing cleaning operations with ReAct agent...")
        
        task_description = f"""
You have access to data cleaning tools. Your task is to clean the dataset based on this plan:

{cleaning_plan}

Rationale: {rationale}

The dataset currently has {original_shape[0]} rows and {original_shape[1]} columns.

Available tools:
- handle_missing_values(strategy, numeric_strategy, categorical_strategy, columns): Handle missing values. Use strategy for single approach or separate numeric_strategy/categorical_strategy for type-specific handling. Examples: numeric_strategy='mean', categorical_strategy='most_frequent'
- remove_duplicates(subset, keep): Remove duplicate rows
- remove_outliers(columns, method, threshold): Remove outliers
- clean_text_columns(columns, operations): Clean text data
- convert_data_types(type_conversions): Convert column types
- inspect_data(columns): Inspect current data state
- get_operations_log(): See what operations have been performed

Execute the cleaning plan step by step. Inspect the data first, then perform each cleaning operation as needed.
"""
        
        # Execute with ReAct
        react_result = self.react_agent(task = task_description)
        
        print(f"\n🤖 ReAct Agent Result:")
        print(react_result)
        
        # Get the final cleaned dataframe
        final_df = self.cleaning_tool.get_current_dataframe()
        final_shape = final_df.shape
        
        # Save the cleaned dataset
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_cleaned{input_path_obj.suffix}")
        
        print(f"\n💾 Saving cleaned dataset to: {output_path}")
        self.io_tool.write_spreadsheet(final_df, output_path)
        
        # Generate cleaning report
        report_path = output_path.replace('.csv', '_report.md').replace('.xlsx', '_report.md')
        self.generate_report(input_path, output_path, original_shape, final_shape, 
                           cleaning_plan, rationale, react_result, report_path)
        
        result = {
            'input_file': input_path,
            'output_file': output_path,
            'report_file': report_path,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'cleaning_plan': cleaning_plan,
            'rationale': rationale,
            'react_result': react_result,
            'operations_log': self.cleaning_tool.operations_log
        }
        
        print(f"\n✅ Cleaning completed!")
        print(f"📈 Result: {original_shape[0]} → {final_shape[0]} rows, {original_shape[1]} → {final_shape[1]} columns")
        print(f"📄 Report saved to: {report_path}")
        
        return result
    
    def generate_report(self, input_path: str, output_path: str, original_shape: tuple, 
                       final_shape: tuple, cleaning_plan: str, rationale: str, 
                       react_result: str, report_path: str):
        """Generate a markdown report of the cleaning process"""
        
        report_content = f"""# Data Cleaning Report

## Summary
- **Input File**: `{input_path}`
- **Output File**: `{output_path}`
- **Original Shape**: {original_shape[0]} rows × {original_shape[1]} columns
- **Final Shape**: {final_shape[0]} rows × {final_shape[1]} columns
- **Rows Changed**: {original_shape[0] - final_shape[0]}
- **Columns Changed**: {original_shape[1] - final_shape[1]}

## Cleaning Plan
{cleaning_plan}

## Rationale
{rationale}

## ReAct Agent Execution
{react_result}

## Operations Log
"""
        
        for i, op in enumerate(self.cleaning_tool.operations_log, 1):
            report_content += f"""
### {i}. {op['operation']}
- **Parameters**: `{op['parameters']}`
- **Impact**: {op['rows_before']} → {op['rows_after']} rows
"""
        
        self.io_tool.write_markdown(report_content, report_path, title="Data Cleaning Report")
