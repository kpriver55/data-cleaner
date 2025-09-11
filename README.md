# DSPy Data Cleaning Agent

A sophisticated data cleaning agent built using DSPy (Declarative Self-improving Python) that leverages Large Language Models (LLMs) to intelligently analyze, plan, and execute data cleaning operations on messy datasets.

## 🚀 Features

- **Intelligent Data Analysis**: Uses LLM reasoning to analyze data quality and identify cleaning needs
- **Automated Cleaning Plan Generation**: Creates step-by-step cleaning plans with rationale
- **ReAct Agent Execution**: Implements ReAct (Reasoning + Acting) pattern for tool-based execution
- **Comprehensive Data Operations**:
  - Missing value handling (mean, median, mode, forward/back fill, KNN imputation)
  - Duplicate removal
  - Outlier detection and removal
  - Text data cleaning (whitespace, case normalization, special characters)
  - Data type conversions
  - Statistical analysis and reporting

- **Detailed Reporting**: Generates comprehensive markdown reports of all cleaning operations
- **Extensible Architecture**: Modular design with separate tools for different operations

## 📁 Project Structure

```
data-cleaner/
├── main.py                        # Main entry point with demo
├── data_cleaning_agent.py          # Core DSPy agent implementation
├── data_cleaning_tool.py           # ReAct-compatible tool wrapper
├── data_transformation_tool.py     # Data transformation operations
├── file_io_tool.py                 # File I/O operations
├── stats_tool.py                   # Statistical analysis tools
├── signatures.py                   # DSPy signatures for LLM interactions
└── requirements.txt                # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data-cleaner
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama** (for local LLM):
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama pull llama3.2:3b-instruct-q6_k
   ollama serve  # Start Ollama server on localhost:11434
   ```

## 🔧 Usage

### Basic Usage

```python
from data_cleaning_agent import DataCleaningAgent
from file_io_tool import FileIOTool
from stats_tool import StatisticalAnalysisTool
from data_transformation_tool import DataTransformationTool

# Initialize tools
io_tool = FileIOTool()
stats_tool = StatisticalAnalysisTool()
transform_tool = DataTransformationTool()

# Create agent
agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)

# Clean a dataset
result = agent.clean_dataset('your_messy_data.csv')
```

### Running the Demo

```bash
python main.py
```

This will:
1. Generate sample messy data with various quality issues
2. Analyze the data using the LLM
3. Create an intelligent cleaning plan
4. Execute the plan using the ReAct agent
5. Save cleaned data and generate a detailed report

## 🧠 How It Works

### 1. Data Analysis Phase
- Loads and analyzes the input dataset
- Identifies missing values, data types, outliers, and quality issues
- Generates comprehensive statistical summaries

### 2. LLM Planning Phase
- Uses DSPy's ChainOfThought with custom signatures
- Analyzes data quality metrics and sample data
- Generates a step-by-step cleaning plan with rationale

### 3. ReAct Execution Phase
- Implements ReAct (Reasoning + Acting) pattern
- Uses available tools to execute the cleaning plan
- Provides real-time feedback and adjustments

### 4. Reporting Phase
- Tracks all operations performed
- Generates detailed markdown reports
- Saves cleaned datasets with proper naming

## 🔧 Configuration

### LLM Configuration

The agent is configured to use Ollama with Llama 3.2. You can modify the LLM configuration in `main.py`:

```python
# Configure DSPy to use different LLM
ollama_lm = dspy.LM("ollama_chat/llama3.2:3b-instruct-q6_k", 
                    api_base="http://localhost:11434", 
                    api_key="")
dspy.settings.configure(lm=ollama_lm)
```

### Tool Configuration

Each tool can be configured with custom parameters:

```python
# Custom missing value handling
agent.cleaning_tool.handle_missing_values(strategy='knn', columns=['age', 'income'])

# Custom outlier removal
agent.cleaning_tool.remove_outliers(columns=['salary'], method='iqr', threshold=1.5)
```

## 📊 Example Output

The agent generates detailed reports showing:

- **Before/After Statistics**: Dataset shape changes, missing value counts
- **Cleaning Plan**: Step-by-step operations with rationale
- **ReAct Execution Log**: Detailed reasoning and tool usage
- **Operations Log**: Precise record of all transformations performed

## 🔗 Dependencies

### Core Dependencies
- **dspy**: DSPy framework for LLM programming
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **scipy**: Scientific computing

### LLM Dependencies
- **litellm**: LLM API abstraction
- **openai**: OpenAI API client
- **httpx**: HTTP client for API calls

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy) framework
- Uses [Ollama](https://ollama.ai) for local LLM inference
- Inspired by ReAct (Reasoning + Acting) paradigm for AI agents

## 🔍 Example Use Cases

- **Data Science Preprocessing**: Clean datasets before analysis or modeling
- **Business Intelligence**: Prepare data for reporting and dashboards  
- **Research Data**: Clean experimental or survey data
- **ETL Pipelines**: Automated data quality assurance
- **Data Migration**: Clean data during system migrations

## ⚠️ Known Limitations

- Requires Ollama server running locally
- Performance depends on LLM model size and capabilities
- Large datasets may require chunking for memory efficiency
- Complex domain-specific cleaning may require custom rules