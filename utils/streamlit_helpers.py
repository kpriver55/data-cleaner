"""
Shared Streamlit Helper Functions

Common utilities used across multiple pages in the Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


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


def display_data_overview(df, title: Optional[str] = "Data Overview"):
    """Display overview of dataframe"""
    if title:
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
            'Missing Count': int(missing_count),
            'Missing %': f"{missing_pct:.1f}%",
            'Unique Values': int(df[col].nunique())
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
                st.write("Missing Values by Column:")
                st.write(dict(zip(column_names, missing_values)))
        else:
            st.info("No missing data found!")

    with col2:
        # Data types distribution
        try:
            dtype_counts = df.dtypes.value_counts()
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
            st.write("Data Types Distribution:")
            for dtype, count in zip(dtype_names, dtype_values):
                st.write(f"- {dtype}: {count} columns")

    # Numeric columns distribution
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.subheader("Numeric Columns Distribution")

        num_cols = min(len(numeric_cols), 4)
        if num_cols > 0:
            selected_numeric = st.multiselect(
                "Select numeric columns to visualize:",
                numeric_cols.tolist(),
                default=numeric_cols.tolist()[:4]
            )

            if selected_numeric:
                df_plotly = prepare_df_for_plotly(df)

                for col in selected_numeric:
                    try:
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
                        st.write(f"Basic statistics for {col}:")
                        try:
                            st.write(df[col].describe())
                        except:
                            st.write(f"Unable to display statistics for {col}")


def initialize_session_state():
    """Initialize session state variables if not already set"""
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
    if 'llm_config' not in st.session_state:
        from llm_config import LLMConfig
        st.session_state.llm_config = LLMConfig()
