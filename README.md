# 🧹 AI Data Cleaning Assistant

An intelligent **web-based data cleaning application** built with Streamlit that leverages Large Language Models (LLMs) to automatically analyze, plan, and execute data cleaning operations. Upload your messy datasets and let AI clean them for you through an intuitive web interface.

## ✨ Key Features

### 🌐 Web Interface (Primary Usage)
- **Upload & Clean**: Drag-and-drop CSV/Excel files directly in your browser
- **AI-Powered Analysis**: Automatic data quality assessment with visual insights
- **Interactive Visualizations**: Real-time charts showing data quality issues and cleaning progress
- **Smart Cleaning Plans**: AI generates step-by-step cleaning strategies with explanations
- **Multiple Export Options**: Download cleaned data as CSV, Excel, or with detailed reports
- **Model Selection**: Choose from multiple LLM models for different cleaning approaches
- **Real-time Progress**: Watch the AI work through cleaning operations with live updates

### 🤖 AI Capabilities
- **Intelligent Data Analysis**: Uses LLM reasoning to identify data quality issues
- **Automated Planning**: Creates comprehensive cleaning plans with detailed rationale
- **ReAct Agent Execution**: Implements Reasoning + Acting pattern for systematic cleaning
- **Comprehensive Operations**:
  - Missing value handling (mean, median, mode, forward/back fill, KNN imputation)
  - Duplicate detection and removal
  - Outlier detection and handling
  - Text data normalization (whitespace, case, special characters)
  - Smart data type conversions
  - Statistical analysis and validation
- **Detailed Reporting**: Generates comprehensive markdown reports of all operations performed

## 📁 Project Structure

```
data-cleaner/
├── app.py                          # 🌐 Streamlit web application (PRIMARY INTERFACE)
├── main.py                         # 🔧 Command-line demo script
├── data_cleaning_agent.py          # 🤖 Core DSPy agent implementation
├── data_cleaning_tool.py           # ⚙️  ReAct-compatible tool wrapper
├── data_transformation_tool.py     # 🔄 Data transformation operations
├── file_io_tool.py                 # 📁 File I/O operations
├── stats_tool.py                   # 📊 Statistical analysis tools
├── signatures.py                   # 📝 DSPy signatures for LLM interactions
└── requirements.txt                # 📦 Python dependencies
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd data-cleaner

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM Provider

The app supports both **local** (Ollama) and **cloud-based** LLM providers.

#### Option A: Local LLM (Ollama) - Free & Private
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:7b-instruct-q5_k_m
ollama serve  # Runs on localhost:11434
```

#### Option B: Cloud API Providers
Set environment variables for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Together AI
export TOGETHER_API_KEY="..."

# Anyscale
export ANYSCALE_API_KEY="..."
```

Or create a `.env` file (see `.env.example`):
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

### 3. Launch the Web App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🖥️ How to Use

### Web Interface (Recommended)

1. **Launch the app**: `streamlit run app.py`
2. **Upload your data**: Drag and drop CSV or Excel files
3. **Configure settings**: Choose your preferred LLM model in the sidebar
4. **Initialize AI agent**: Click "Initialize AI Agent"
5. **Review data**: Explore the "Data Overview" tab to understand your dataset
6. **Start cleaning**: Go to "AI Cleaning" tab and click "Start AI Cleaning"
7. **Review results**: Check the "Results" tab and download cleaned data

### Features in the Web App
- **📊 Data Overview**: Visualize data quality issues, missing values, and data types
- **🤖 AI Cleaning**: Watch the AI analyze, plan, and execute cleaning operations
- **📈 Results**: Compare before/after statistics and download cleaned datasets
- **⚙️ Model Selection**: Choose from multiple LLM models (Qwen, Llama, etc.)
- **🔧 Cleaning Options**: Configure conservative mode and auto-execution settings

### Command Line (Alternative)

For programmatic usage or testing:
```bash
python main.py  # Run demo with sample data
```

## 🧠 How It Works

### 1. 📊 Smart Data Analysis
- **Upload Detection**: Automatically detects file format and loads data
- **Quality Assessment**: AI analyzes missing values, data types, outliers, and inconsistencies
- **Visual Insights**: Interactive charts reveal data quality patterns
- **Statistical Summary**: Comprehensive overview of dataset characteristics

### 2. 🤖 AI Planning
- **LLM Analysis**: Uses DSPy framework with custom reasoning signatures
- **Context Understanding**: AI examines data samples and quality metrics
- **Smart Strategy**: Generates step-by-step cleaning plan with detailed rationale
- **Risk Assessment**: Identifies potential issues and suggests conservative approaches

### 3. ⚡ Automated Execution
- **ReAct Pattern**: Implements Reasoning + Acting for systematic cleaning
- **Tool Orchestration**: Coordinates multiple specialized cleaning tools
- **Real-time Updates**: Live progress tracking in the web interface
- **Error Handling**: Graceful handling of edge cases and data anomalies

### 4. 📋 Results & Reporting
- **Before/After Comparison**: Visual comparison of dataset improvements
- **Operation Log**: Detailed record of all transformations performed
- **Quality Metrics**: Statistical validation of cleaning effectiveness
- **Export Options**: Multiple download formats with comprehensive reports

## ⚙️ Configuration

### 🤖 LLM Provider Options

The app supports multiple LLM providers configurable via the web UI or code:

| Provider | Type | Configuration | Best For |
|----------|------|---------------|----------|
| **Ollama** | Local | Default, no API key needed | Privacy, no cost, offline |
| **OpenAI** | Cloud API | Set `OPENAI_API_KEY` | Ease of use, quality |
| **Anthropic Claude** | Cloud API | Set `ANTHROPIC_API_KEY` | Complex reasoning |
| **Together AI** | Cloud API | Set `TOGETHER_API_KEY` | Fast inference |
| **Anyscale** | Cloud API | Set `ANYSCALE_API_KEY` | Enterprise deployment |

#### Web UI Configuration
1. Select provider in sidebar
2. Choose model
3. Enter API key (or set via environment variable)
4. Click "Initialize AI Agent"

#### Programmatic Configuration
```python
from llm_config import setup_llm

# Local Ollama (default)
setup_llm("ollama", "qwen2.5:7b-instruct-q5_k_m")

# OpenAI
setup_llm("openai", "gpt-4o-mini")  # Reads OPENAI_API_KEY from env

# Anthropic Claude
setup_llm("anthropic", "claude-3-5-sonnet-20241022")

# Environment-based (reads LLM_PROVIDER and LLM_MODEL)
setup_llm()
```

#### Recommended Models
- **Ollama**: `qwen2.5:7b-instruct-q5_k_m` (best balance)
- **OpenAI**: `gpt-4o-mini` (cost-effective) or `gpt-4o` (highest quality)
- **Anthropic**: `claude-3-5-sonnet-20241022` (complex tasks) or `claude-3-5-haiku-20241022` (fast)

### 🔧 Cleaning Options
Configure in the sidebar:
- **Auto-execute high confidence operations**: Automatically apply safe transformations
- **Conservative mode**: Prefer safer operations over aggressive cleaning

### 🔐 Security Best Practices
- ✅ **DO**: Store API keys in environment variables or `.env` files
- ✅ **DO**: Add `.env` to `.gitignore` (already configured)
- ❌ **DON'T**: Hardcode API keys in source code
- ❌ **DON'T**: Commit `.env` files to version control

See `.env.example` for configuration template.

## 📊 What You Get

### 🖥️ Interactive Web Interface
- **Real-time Visualizations**: Data quality charts, missing value heatmaps, distribution plots
- **Progress Tracking**: Live updates as AI analyzes and cleans your data
- **Before/After Comparison**: Side-by-side statistics showing improvements
- **AI Reasoning Display**: See exactly how the AI analyzes your data and makes decisions

### 📋 Comprehensive Reports
- **Cleaning Strategy**: AI-generated plan with detailed rationale for each operation
- **Execution Log**: Step-by-step record of all transformations performed
- **Quality Metrics**: Statistical validation showing data improvement
- **Operation Summary**: Clear breakdown of rows affected, changes made, and time taken

### 💾 Multiple Export Options
- **CSV Format**: Universal compatibility for further analysis
- **Excel Format**: Formatted spreadsheet with preserved data types
- **Markdown Report**: Detailed documentation of the entire cleaning process

## 📦 Dependencies

### 🌐 Web Application
- **streamlit**: Modern web app framework for the user interface
- **plotly**: Interactive data visualizations and charts

### 🤖 AI & Data Processing
- **dspy**: DSPy framework for LLM programming and agent orchestration
- **pandas**: Data manipulation and analysis engine
- **numpy**: Numerical computing foundation
- **scikit-learn**: Machine learning utilities for advanced cleaning
- **scipy**: Scientific computing for statistical operations

### 📁 File Support
- **openpyxl**: Excel file reading and writing
- **PyYAML**: Configuration and structured data support

See `requirements.txt` for complete dependency list with version specifications.

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

## 🎯 Perfect For

### 📊 Data Scientists & Analysts
- **Preprocessing**: Clean messy datasets before analysis or machine learning
- **Exploratory Analysis**: Quickly understand and improve data quality
- **Feature Engineering**: Prepare clean features for modeling

### 💼 Business Users
- **Report Preparation**: Clean data for dashboards and business intelligence
- **Data Migration**: Ensure data quality during system transitions
- **Compliance**: Standardize data formats for regulatory requirements

### 🔬 Researchers
- **Survey Data**: Clean and standardize questionnaire responses
- **Experimental Data**: Handle missing values and outliers in research datasets
- **Publication Ready**: Prepare clean datasets for academic publications

### 🏢 Organizations
- **ETL Pipelines**: Automated data quality assurance in data workflows
- **Master Data Management**: Maintain clean, consistent organizational data
- **Data Governance**: Implement systematic data quality improvements

## ⚠️ Requirements & Limitations

### 🔧 System Requirements
- **Python**: 3.8+ with pip package management
- **Memory**: 8GB+ RAM recommended for larger datasets
- **Browser**: Modern web browser with JavaScript enabled
- **LLM Provider**: Either:
  - Ollama server running locally (free, private), OR
  - API key for cloud provider (OpenAI, Anthropic, etc.)

### 📊 Data Limitations
- **File Size**: Very large datasets (>100MB) may require chunking
- **Model Performance**: Cleaning quality depends on chosen LLM model capabilities
- **Domain Specific**: Complex industry-specific rules may need manual configuration