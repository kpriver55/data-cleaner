import dspy


class DataAnalysisSignature(dspy.Signature):
    """Analyze comprehensive data quality issues and create a complete cleaning strategy covering duplicates, missing values, outliers, text issues, and data type problems"""
    comprehensive_data_analysis = dspy.InputField(desc="Complete data quality analysis including dataset info, duplicates, missing values, data types, column statistics, and sample data")

    duplicate_cleaning_plan = dspy.OutputField(desc="Specific steps for handling duplicate rows, if any duplicates are found")
    missing_values_plan = dspy.OutputField(desc="Specific steps for handling missing values, if any missing data is found")
    outlier_handling_plan = dspy.OutputField(desc="Specific steps for handling outliers in numeric columns, if outliers are detected")
    text_cleaning_plan = dspy.OutputField(desc="Specific steps for cleaning text columns, if text quality issues are identified")
    data_type_plan = dspy.OutputField(desc="Specific steps for converting data types, if type conversion issues are identified")
    overall_cleaning_plan = dspy.OutputField(desc="Complete step-by-step cleaning plan combining all necessary operations in logical order")
    rationale = dspy.OutputField(desc="Explanation of why each cleaning step is necessary and the order of operations")

class DataCleaningExecutionSignature(dspy.Signature):
    task = dspy.InputField(desc="step-by-step plan for cleaning data, with specific operations")
    result = dspy.OutputField(desc = "summary of the operations performed")