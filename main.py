import dspy
import scipy
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from file_io_tool import FileIOTool
from stats_tool import StatisticalAnalysisTool
from data_transformation_tool import DataTransformationTool
from data_cleaning_agent import DataCleaningAgent

# Configure DSPy to use Ollama
ollama_lm = dspy.LM("ollama_chat/llama3.2:3b-instruct-q6_k", api_base="http://localhost:11434", api_key="")
dspy.settings.configure(lm=ollama_lm)


    
print("=== DSPy Data Cleaning Agent Demo ===")
print("This agent uses LLM reasoning to clean datasets")
print("Initialize with: agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)")
print("Use with: result = agent.clean_dataset('input.csv')")
    
# Example of creating test data
import pandas as pd
import numpy as np
    
# Create messy test data
np.random.seed(42)
messy_data = pd.DataFrame({
    'id': list(range(1, 101)) + [50, 51, 52],  # Some duplicates
    'name': [f'Person {i}' if i % 10 != 0 else None for i in range(103)],  # Some missing
    'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(103)],
   'salary': np.concatenate([
            np.random.normal(50000, 15000, 98),
            np.random.normal(500000, 600000, 5)  # Some outliers
        ]),
    'email': [f'  person{i}@COMPANY.COM  ' for i in range(103)],  # Needs text cleaning
    'score': ['85.5' if i % 2 == 0 else '90.0' for i in range(103)]  # Should be numeric
    })
messy_data.to_csv('test_messy_data.csv')
io_tool = FileIOTool()
stats_tool = StatisticalAnalysisTool()
transform_tool = DataTransformationTool()
agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)
result = agent.clean_dataset('test_messy_data.csv')