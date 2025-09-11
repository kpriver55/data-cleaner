import dspy


class DataAnalysisSignature(dspy.Signature):
    """Analyze data quality and identify cleaning needs"""
    data_summary = dspy.InputField(desc="Summary statistics and data info from the dataset")
    missing_data_info = dspy.InputField(desc="Information about missing values in the dataset")
    data_types_info = dspy.InputField(desc="Current data types of all columns")
    
    cleaning_plan = dspy.OutputField(desc="Step-by-step plan for cleaning the data, with specific operations")
    rationale = dspy.OutputField(desc="Explanation of why each cleaning step is necessary")

class DataCleaningExecutionSignature(dspy.Signature):
    task = dspy.InputField(desc="step-by-step plan for cleaning data, with specific operations")
    result = dspy.OutputField(desc = "summary of the operations performed")