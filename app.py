"""
Streamlit App for DSPy Data Cleaning Agent
Interactive web interface for cleaning datasets
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import tempfile
from pathlib import Path
import json
import dspy
import numpy as np

# Import your tools and agent
from file_io_tool import FileIOTool
from stats_tool import StatisticalAnalysisTool
from data_transformation_tool import DataTransformationTool
from data_cleaning_agent import DataCleaningAgent
from signatures import DataAnalysisSignature, DataCleaningExecutionSignature
from data_cleaning_tool import DataCleaningTool


# Configure Streamlit page
st.set_page_config(
    page_title="AI Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'cleaning_agent' not in st.session_state:
    st.session_state.cleaning_agent = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'cleaning_results' not in st.session_state:
    st.session_state.cleaning_results = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None

def safe_convert_for_plotly(series_or_value):
    """
    Convert numpy data types to native Python types for Plotly compatibility
    """
    if isinstance(series_or_value, pd.Series):
        return series_or_value.astype(object).apply(lambda x: x.item() if hasattr(x, 'item') else x)
    elif hasattr(series_or_value, 'item'):
        return series_or_value.item()
    elif isinstance(series_or_value, np.ndarray):
        return series_or_value.tolist()
    else:
        return series_or_value

def prepare_df_for_plotly(df):
    """
    Prepare DataFrame for Plotly by converting numpy types to native Python types
    """
    df_copy = df.copy()
    
    # Convert numpy dtypes to native Python types
    for col in df_copy.columns:
        if df_copy[col].dtype.kind in ['i', 'u']:  # integer types
            df_copy[col] = df_copy[col].astype('Int64')  # pandas nullable integer
        elif df_copy[col].dtype.kind in ['f']:  # float types
            df_copy[col] = df_copy[col].astype('float64')
        elif df_copy[col].dtype.kind in ['O']:  # object types
            # Convert any numpy scalars in object columns
            df_copy[col] = df_copy[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    return df_copy

def initialize_agent():
    """Initialize the data cleaning agent"""
    if st.session_state.cleaning_agent is None:
        with st.spinner("Initializing AI cleaning agent..."):
            io_tool = FileIOTool()
            stats_tool = StatisticalAnalysisTool()
            transform_tool = DataTransformationTool()
            st.session_state.cleaning_agent = DataCleaningAgent(io_tool, stats_tool, transform_tool)
        st.success("✅ AI agent initialized!")

def display_data_overview(df, title="Data Overview"):
    """Display overview of dataframe"""
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Data types summary
    st.subheader("Column Information")
    col_info = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        col_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Missing Count': int(missing_count),  # Convert to native int
            'Missing %': f"{missing_pct:.1f}%",
            'Unique Values': int(df[col].nunique())  # Convert to native int
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)

def create_data_quality_charts(df):
    """Create visualizations for data quality"""
    st.subheader("Data Quality Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing data heatmap
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            # Convert to native Python types for Plotly
            missing_values = [int(val) for val in missing_data.values]
            column_names = [str(col) for col in missing_data.index]
            
            try:
                fig = px.bar(
                    x=column_names, 
                    y=missing_values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating missing values chart: {str(e)}")
                # Fallback to simple table
                st.write("Missing Values by Column:")
                st.write(dict(zip(column_names, missing_values)))
        else:
            st.info("No missing data found!")
    
    with col2:
        # Data types distribution
        try:
            dtype_counts = df.dtypes.value_counts()
            # Convert to native Python types
            dtype_names = [str(dtype) for dtype in dtype_counts.index]
            dtype_values = [int(count) for count in dtype_counts.values]
            
            fig = px.pie(
                values=dtype_values,
                names=dtype_names,
                title="Data Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating data types chart: {str(e)}")
            # Fallback to simple display
            st.write("Data Types Distribution:")
            for dtype, count in zip(dtype_names, dtype_values):
                st.write(f"- {dtype}: {count} columns")
    
    # Numeric columns distribution
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.subheader("Numeric Columns Distribution")
        
        # Create subplot for distributions
        num_cols = min(len(numeric_cols), 4)
        if num_cols > 0:
            selected_numeric = st.multiselect(
                "Select numeric columns to visualize:",
                numeric_cols.tolist(),
                default=numeric_cols.tolist()[:4]
            )
            
            if selected_numeric:
                # Prepare data for Plotly
                df_plotly = prepare_df_for_plotly(df)
                
                for col in selected_numeric:
                    try:
                        # Remove any NaN values for the histogram
                        col_data = df_plotly[col].dropna()
                        
                        if len(col_data) > 0:
                            fig = px.histogram(
                                x=col_data, 
                                title=f"Distribution of {col}",
                                labels={'x': col, 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No valid data to plot for column: {col}")
                    except Exception as e:
                        st.error(f"Error creating histogram for {col}: {str(e)}")
                        # Fallback to basic statistics
                        st.write(f"Basic statistics for {col}:")
                        try:
                            st.write(df[col].describe())
                        except:
                            st.write(f"Unable to display statistics for {col}")

def main():
    st.title("🧹 AI Data Cleaning Assistant")
    st.markdown("Upload your data and let AI clean it for you!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection (if you want to make it configurable)
        model_choice = st.selectbox(
            "Select Model",
            ["qwen2.5:7b-instruct-q5_k_m", "llama3.1:8b-instruct-q5_k_M", "qwen2.5:14b-instruct-q4_k_m", "llama3.2:3b-instruct-q6_k"],
            help="Choose the LLM model for data cleaning decisions"
        )

        # Model performance warning
        if "3b" in model_choice.lower():
            st.warning("⚠️ **Performance Note**: 3B parameter models may have difficulty with complex multi-step reasoning required for comprehensive data cleaning. For best results, use 7B+ parameter models.")
        elif "7b" in model_choice.lower() or "8b" in model_choice.lower():
            st.info("✅ **Good Choice**: 7B-8B models provide excellent balance of performance and resource usage for data cleaning tasks.")
        elif "14b" in model_choice.lower():
            st.success("🚀 **Excellent Choice**: 14B+ models provide superior reasoning capabilities for complex data cleaning scenarios.")
        
        # Cleaning options
        st.header("Cleaning Options")
        auto_execute = st.checkbox("Auto-execute high confidence operations", value=True)
        conservative_mode = st.checkbox("Conservative mode (safer operations)", value=True)
        
        # Initialize agent
        if st.button("Initialize AI Agent"):
            try:
                ollama_lm = dspy.LM("ollama_chat/" + model_choice, api_base="http://localhost:11434", api_key="")
                dspy.settings.configure(lm=ollama_lm)
                initialize_agent()
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "📊 Data Overview", "🤖 AI Cleaning", "📈 Results"])
    
    with tab1:
        st.header("Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to begin cleaning"
        )
        
        if uploaded_file is not None:
            try:
                # Store the original filename
                st.session_state.original_filename = uploaded_file.name
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Load the data
                with st.spinner("Loading data..."):
                    io_tool = FileIOTool()
                    df = io_tool.read_spreadsheet(tmp_path)
                    
                    # Clean the dataframe immediately after loading
                    # Convert any problematic dtypes
                    for col in df.columns:
                        if df[col].dtype.name.startswith('int') and df[col].dtype != 'int64':
                            df[col] = df[col].astype('int64')
                        elif df[col].dtype.name.startswith('float') and df[col].dtype != 'float64':
                            df[col] = df[col].astype('float64')
                    
                    st.session_state.original_df = df
                
                st.success(f"✅ Data loaded successfully! {len(df)} rows × {len(df.columns)} columns")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.error("Please ensure your file is a valid CSV or Excel file.")
    
    with tab2:
        if st.session_state.original_df is not None:
            try:
                display_data_overview(st.session_state.original_df, "Original Data Overview")
                create_data_quality_charts(st.session_state.original_df)
            except Exception as e:
                st.error(f"Error displaying data overview: {str(e)}")
                # Fallback: show basic info
                st.write("Basic Data Information:")
                st.write(f"Shape: {st.session_state.original_df.shape}")
                st.write("Columns:", list(st.session_state.original_df.columns))
                st.write("Data Types:")
                st.write(st.session_state.original_df.dtypes)
        else:
            st.info("Please upload a dataset first.")
    
    with tab3:
        if st.session_state.original_df is not None:
            st.header("🤖 AI-Powered Data Cleaning")
            
            if st.session_state.cleaning_agent is None:
                st.warning("Please initialize the AI agent first (see sidebar)")
                if st.button("Initialize AI Agent Now"):
                    initialize_agent()
            else:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("The AI will analyze your data and create a cleaning plan.")
                
                with col2:
                    start_cleaning = st.button("🚀 Start AI Cleaning", type="primary")
                
                if start_cleaning:
                    with st.spinner("AI is analyzing your data..."):
                        # Create temporary file for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            st.session_state.original_df.to_csv(tmp_file.name, index=False)
                            tmp_path = tmp_file.name
                        
                        try:
                            # Run the cleaning agent
                            results = st.session_state.cleaning_agent.clean_dataset(tmp_path)
                            st.session_state.cleaning_results = results
                            
                            # Load the cleaned data
                            cleaned_df = st.session_state.cleaning_agent.io_tool.read_spreadsheet(results['output_file'])
                            st.session_state.cleaned_df = cleaned_df
                            
                            st.success("🎉 Data cleaning completed!")
                            
                            # Show cleaning summary
                            st.subheader("Cleaning Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Rows", 
                                    f"{results['final_shape'][0]:,}",
                                    delta=f"{results['final_shape'][0] - results['original_shape'][0]:,}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Columns", 
                                    results['final_shape'][1],
                                    delta=results['final_shape'][1] - results['original_shape'][1]
                                )
                            
                            with col3:
                                operations_count = len(results['operations_log'])
                                st.metric("Operations Performed", operations_count)
                            
                            # Show cleaning plan and rationale
                            with st.expander("🧠 AI Reasoning and Plan"):
                                st.subheader("Cleaning Plan")
                                st.write(results['cleaning_plan'])
                                
                                st.subheader("Rationale")
                                st.write(results['rationale'])
                                
                                if 'react_result' in results:
                                    st.subheader("Execution Details")
                                    st.code(results['react_result'])
                            
                            # Show operations log
                            if results['operations_log']:
                                with st.expander("📋 Operations Performed"):
                                    for i, op in enumerate(results['operations_log'], 1):
                                        st.write(f"**{i}. {op['operation']}**")
                                        st.write(f"Parameters: `{op['parameters']}`")
                                        st.write(f"Impact: {op['rows_before']} → {op['rows_after']} rows")
                                        st.write("---")
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Error during cleaning: {str(e)}")
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
        else:
            st.info("Please upload a dataset first.")
    
    with tab4:
        if st.session_state.cleaned_df is not None:
            st.header("📊 Cleaning Results")
            
            # Before/After comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Before Cleaning")
                try:
                    display_data_overview(st.session_state.original_df, "")
                except Exception as e:
                    st.error(f"Error displaying original data overview: {str(e)}")
            
            with col2:
                st.subheader("After Cleaning")
                try:
                    display_data_overview(st.session_state.cleaned_df, "")
                except Exception as e:
                    st.error(f"Error displaying cleaned data overview: {str(e)}")
            
            # Show cleaned data
            st.subheader("Cleaned Dataset Preview")
            st.dataframe(st.session_state.cleaned_df.head(20), use_container_width=True)
            
            # Download options
            st.subheader("💾 Download Cleaned Data")
            
            # Generate filenames based on original file
            base_name = Path(st.session_state.original_filename).stem if st.session_state.original_filename else "cleaned_data"
            csv_filename = f"{base_name}_cleaned.csv"
            excel_filename = f"{base_name}_cleaned.xlsx"
            report_filename = f"{base_name}_cleaning_report.md"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download as CSV
                try:
                    csv_data = st.session_state.cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=csv_filename,
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error preparing CSV download: {str(e)}")
            
            with col2:
                # Download as Excel
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        st.session_state.cleaned_df.to_excel(writer, index=False)
                    excel_data = buffer.getvalue()
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error preparing Excel download: {str(e)}")
            
            with col3:
                # Download cleaning report
                try:
                    if st.session_state.cleaning_results and 'report_file' in st.session_state.cleaning_results:
                        if os.path.exists(st.session_state.cleaning_results['report_file']):
                            with open(st.session_state.cleaning_results['report_file'], 'r') as f:
                                report_content = f.read()
                            
                            st.download_button(
                                label="📋 Download Report",
                                data=report_content,
                                file_name=report_filename,
                                mime="text/markdown"
                            )
                except Exception as e:
                    st.error(f"Error preparing report download: {str(e)}")
        
        elif st.session_state.original_df is not None:
            st.info("Run the AI cleaning process first to see results.")
        else:
            st.info("Please upload a dataset and run the cleaning process.")

if __name__ == "__main__":
    main()