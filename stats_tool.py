"""
DSPy Statistical Analysis Tool for Data Preprocessing Agent
Handles mathematical and statistical operations 
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalysisTool:
    """
    A comprehensive statistical analysis tool for numeric data.
    Provides reliable mathematical computations for data preprocessing tasks.
    """
    
    def __init__(self):
        self.numeric_types = ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']
    
    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify numeric columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of numeric column names
        """
        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        return numeric_cols
    
    def basic_statistics(self, 
                        df: pd.DataFrame, 
                        columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute basic statistics for numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (None for all numeric)
            
        Returns:
            Dictionary with statistics for each column
        """
        if columns is None:
            columns = self.get_numeric_columns(df)
        
        results = {}
        
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame")
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: Column '{col}' is not numeric, skipping")
                continue
            
            series = df[col].dropna()  # Remove NaN values
            
            if len(series) == 0:
                results[col] = {"error": "No valid numeric values"}
                continue
            
            stats_dict = {
                "count": len(series),
                "missing_count": df[col].isna().sum(),
                "missing_percentage": round((df[col].isna().sum() / len(df)) * 100, 2),
                "mean": round(float(series.mean()), 6),
                "median": round(float(series.median()), 6),
                "mode": float(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                "std": round(float(series.std()), 6),
                "variance": round(float(series.var()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
                "range": round(float(series.max() - series.min()), 6),
                "q25": round(float(series.quantile(0.25)), 6),
                "q50": round(float(series.quantile(0.50)), 6),  # Same as median
                "q75": round(float(series.quantile(0.75)), 6),
                "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 6),
                "skewness": round(float(series.skew()), 6),
                "kurtosis": round(float(series.kurtosis()), 6)
            }
            
            results[col] = stats_dict
        
        return results
    
    def percentiles(self, 
                   df: pd.DataFrame, 
                   column: str, 
                   percentiles: List[float] = [5, 10, 25, 50, 75, 90, 95, 99]) -> Dict[str, float]:
        """
        Calculate specific percentiles for a column.
        
        Args:
            df: Input DataFrame
            column: Column name
            percentiles: List of percentiles to calculate (0-100)
            
        Returns:
            Dictionary of percentile values
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not numeric"}
        
        series = df[column].dropna()
        
        if len(series) == 0:
            return {"error": "No valid numeric values"}
        
        results = {}
        for p in percentiles:
            if 0 <= p <= 100:
                results[f"p{p}"] = round(float(series.quantile(p/100)), 6)
            else:
                print(f"Warning: Percentile {p} is out of range [0-100], skipping")
        
        return results
    
    def correlation_matrix(self, 
                          df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix for numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (None for all numeric)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix as DataFrame
        """
        if columns is None:
            columns = self.get_numeric_columns(df)
        
        if len(columns) < 2:
            print("Warning: Need at least 2 numeric columns for correlation")
            return pd.DataFrame()
        
        try:
            corr_df = df[columns].corr(method=method)
            return corr_df.round(6)
        except Exception as e:
            print(f"Error computing correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       column: str, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in a numeric column.
        
        Args:
            df: Input DataFrame
            column: Column name
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not numeric"}
        
        series = df[column].dropna()
        
        if len(series) == 0:
            return {"error": "No valid numeric values"}
        
        if method == 'iqr':
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outliers = series[z_scores > threshold]
            lower_bound = series.mean() - threshold * series.std()
            upper_bound = series.mean() + threshold * series.std()
            
        else:
            return {"error": f"Unknown method: {method}"}
        
        return {
            "method": method,
            "threshold": threshold,
            "outlier_count": len(outliers),
            "outlier_percentage": round((len(outliers) / len(series)) * 100, 2),
            "outlier_values": outliers.tolist(),
            "lower_bound": round(float(lower_bound), 6),
            "upper_bound": round(float(upper_bound), 6),
            "outlier_indices": outliers.index.tolist()
        }
    
    def normality_test(self, 
                      df: pd.DataFrame, 
                      column: str) -> Dict[str, Any]:
        """
        Test for normality using Shapiro-Wilk test.
        
        Args:
            df: Input DataFrame
            column: Column name
            
        Returns:
            Dictionary with normality test results
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not numeric"}
        
        series = df[column].dropna()
        
        if len(series) < 3:
            return {"error": "Need at least 3 values for normality test"}
        
        if len(series) > 5000:
            # Sample for large datasets as Shapiro-Wilk has limitations
            series = series.sample(n=5000, random_state=42)
        
        try:
            statistic, p_value = stats.shapiro(series)
            
            return {
                "test": "Shapiro-Wilk",
                "statistic": round(float(statistic), 6),
                "p_value": round(float(p_value), 6),
                "is_normal": p_value > 0.05,
                "interpretation": "Normal distribution" if p_value > 0.05 else "Not normal distribution",
                "sample_size": len(series)
            }
        except Exception as e:
            return {"error": f"Error in normality test: {str(e)}"}
    
    def group_statistics(self, 
                        df: pd.DataFrame, 
                        group_column: str, 
                        value_column: str) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for groups.
        
        Args:
            df: Input DataFrame
            group_column: Column to group by
            value_column: Numeric column to analyze
            
        Returns:
            Dictionary with statistics for each group
        """
        if group_column not in df.columns:
            return {"error": f"Group column '{group_column}' not found"}
        
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return {"error": f"Value column '{value_column}' is not numeric"}
        
        results = {}
        
        for group_name, group_data in df.groupby(group_column):
            series = group_data[value_column].dropna()
            
            if len(series) == 0:
                results[str(group_name)] = {"error": "No valid values"}
                continue
            
            results[str(group_name)] = {
                "count": len(series),
                "mean": round(float(series.mean()), 6),
                "median": round(float(series.median()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
                "q25": round(float(series.quantile(0.25)), 6),
                "q75": round(float(series.quantile(0.75)), 6)
            }
        
        return results
    
    def value_counts(self, 
                    df: pd.DataFrame, 
                    column: str, 
                    normalize: bool = False,
                    top_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Count unique values in a column.
        
        Args:
            df: Input DataFrame
            column: Column name
            normalize: Return proportions instead of counts
            top_n: Return only top N values
            
        Returns:
            Dictionary with value counts
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        try:
            counts = df[column].value_counts(normalize=normalize, dropna=False)
            
            if top_n:
                counts = counts.head(top_n)
            
            return {
                "total_unique": len(df[column].unique()),
                "total_rows": len(df),
                "null_count": df[column].isna().sum(),
                "value_counts": counts.to_dict()
            }
        except Exception as e:
            return {"error": f"Error counting values: {str(e)}"}
    
    def summary_report(self, 
                      df: pd.DataFrame, 
                      columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze
            
        Returns:
            Complete statistical summary
        """
        if columns is None:
            columns = self.get_numeric_columns(df)
        
        report = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2)
            },
            "column_statistics": self.basic_statistics(df, columns),
            "missing_data_summary": {},
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        # Missing data summary
        missing_summary = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_summary[col] = {
                    "count": int(missing_count),
                    "percentage": round((missing_count / len(df)) * 100, 2)
                }
        
        report["missing_data_summary"] = missing_summary
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize the tool
    stats_tool = StatisticalAnalysisTool()
    
    # Create sample data with various patterns
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'normal_dist': np.random.normal(100, 15, 1000),
        'skewed_dist': np.random.exponential(2, 1000),
        'uniform_dist': np.random.uniform(0, 100, 1000),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'with_outliers': np.concatenate([np.random.normal(50, 10, 995), [200, 300, -100, 400, 500]]),
        'with_missing': [np.random.normal(75, 12, 800).tolist() + [None] * 200][0]
    })
    
    print("=== Statistical Analysis Tool Demo ===")
    
    # Basic statistics
    print("\n1. Basic Statistics:")
    basic_stats = stats_tool.basic_statistics(sample_data)
    for col, stats in basic_stats.items():
        if 'error' not in stats:
            print(f"{col}: mean={stats['mean']}, median={stats['median']}, std={stats['std']}")
    
    # Percentiles
    print("\n2. Percentiles for 'normal_dist':")
    percentiles = stats_tool.percentiles(sample_data, 'normal_dist')
    print(percentiles)
    
    # Outlier detection
    print("\n3. Outlier Detection for 'with_outliers':")
    outliers = stats_tool.detect_outliers(sample_data, 'with_outliers')
    print(f"Found {outliers['outlier_count']} outliers ({outliers['outlier_percentage']}%)")
    
    # Group statistics
    print("\n4. Group Statistics:")
    group_stats = stats_tool.group_statistics(sample_data, 'categorical', 'normal_dist')
    for group, stats in group_stats.items():
        if 'error' not in stats:
            print(f"Group {group}: mean={stats['mean']}, count={stats['count']}")
    
    # Summary report
    print("\n5. Summary Report:")
    report = stats_tool.summary_report(sample_data)
    print(f"Dataset: {report['dataset_info']['total_rows']} rows, {report['dataset_info']['numeric_columns']} numeric columns")
    print(f"Missing data in {len(report['missing_data_summary'])} columns")