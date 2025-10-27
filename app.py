"""
AI Data Cleaning Assistant - Home Page

Multi-page Streamlit application for AI-powered data cleaning and optimization.
"""

import streamlit as st

from utils import initialize_session_state

# Configure page
st.set_page_config(
    page_title="AI Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
initialize_session_state()

# Main content
st.title("🧹 AI Data Cleaning Assistant")
st.markdown("### Welcome to your intelligent data cleaning companion!")

st.markdown(
    """
This application uses Large Language Models (LLMs) to automatically analyze, plan,
and execute data cleaning operations on your datasets.

## 🚀 Features

### 📊 Data Cleaning
Upload messy datasets and let AI:
- Analyze data quality issues
- Generate comprehensive cleaning plans
- Execute cleaning operations automatically
- Provide detailed reports

### 🎯 Model Optimization
Improve the AI's performance:
- Train on your specific cleaning tasks
- Optimize prompts with DSPy
- Evaluate and compare models
- Save and deploy optimized agents

## 📍 Getting Started

1. **Configure LLM** - Choose your preferred LLM provider in the sidebar
2. **Upload Data** - Go to the Data Cleaning page and upload your dataset
3. **Clean** - Let the AI analyze and clean your data
4. **Optimize** (Optional) - Train the AI on your examples for better results

## 📄 Pages

Use the sidebar to navigate between:
- **🧹 Data Cleaning** - Clean your datasets
- **🎯 Optimization** - Train and optimize the AI

---

👈 Select a page from the sidebar to begin!
"""
)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
    This app uses:
    - **DSPy** framework for LLM programming
    - **ReAct** pattern for agent reasoning
    - **Multiple LLM providers** (Ollama, OpenAI, Anthropic, etc.)

    ---

    **Quick Links:**
    - [Documentation](https://github.com/anthropics/claude-code)
    - [Report Issues](https://github.com/anthropics/claude-code/issues)
    """
    )

    # Show current LLM configuration if available
    if st.session_state.llm_config and st.session_state.llm_config.lm is not None:
        config = st.session_state.llm_config.get_current_config()
        st.success(f"✅ LLM: {config['provider']}/{config['model']}")
    else:
        st.warning("⚠️ LLM not configured")
        st.info("Configure on Data Cleaning page")
