"""
DSPy Data Transformation Tool for Data Preprocessing Agent
Handles data cleaning and transformation operations while preserving original data
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataTransformationTool:
    """
    Comprehensive data transformation tool for preprocessing tasks.
    All operations return new DataFrames, preserving original data.
    """
    
    def __init__(self):
        self.transformation_log = []
        self.scalers = {}
        self.encoders = {}
    
    def log_transformation(self, operation: str, details: Dict[str, Any]):
        """Log transformation operations for reproducibility."""
        self.transformation_log.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'details': details
        })
    
    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """Return the transformation log."""
        return self.transformation_log
    
    def handle_missing_values(self, 
                             df: pd.DataFrame, 
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None,
                             custom_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies.
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'mode', 'constant', 'drop', 'forward_fill', 'back_fill', 'knn'
            columns: Specific columns to process (None for all)
            custom_values: Custom fill values for 'constant' strategy
            
        Returns:
            New DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        processed_columns = []
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=columns)
            
        elif strategy in ['mean', 'median', 'most_frequent']:
            for col in columns:
                if col in df_clean.columns and df_clean[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        if strategy == 'mean':
                            fill_value = df_clean[col].mean()
                        elif strategy == 'median':
                            fill_value = df_clean[col].median()
                        else:  # most_frequent for numeric
                            fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else df_clean[col].mean()
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        processed_columns.append(col)
                    elif strategy == 'most_frequent':
                        # For categorical columns
                        fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        processed_columns.append(col)
        
        elif strategy == 'constant':
            for col in columns:
                if col in df_clean.columns and df_clean[col].isna().any():
                    fill_value = custom_values.get(col, 0) if custom_values else 0
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    processed_columns.append(col)
        
        elif strategy == 'forward_fill':
            df_clean[columns] = df_clean[columns].fillna(method='ffill')
            processed_columns = columns
            
        elif strategy == 'back_fill':
            df_clean[columns] = df_clean[columns].fillna(method='bfill')
            processed_columns = columns
            
        elif strategy == 'knn':
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df_clean[col])]
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                processed_columns = numeric_cols
        
        self.log_transformation('handle_missing_values', {
            'strategy': strategy,
            'columns_processed': processed_columns,
            'custom_values': custom_values
        })
        
        return df_clean
    
    def remove_outliers(self, 
                       df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove outliers from numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to process (None for all numeric)
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (cleaned DataFrame, outlier report)
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        outlier_report = {}
        total_removed = 0
        
        for col in columns:
            if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            series = df_clean[col].dropna()
            original_count = len(df_clean)
            
            if method == 'iqr':
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - threshold * iqr
                upper_bound = q75 + threshold * iqr
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outlier_mask = z_scores > threshold
                lower_bound = df_clean[col].mean() - threshold * df_clean[col].std()
                upper_bound = df_clean[col].mean() + threshold * df_clean[col].std()
            
            outliers_found = outlier_mask.sum()
            df_clean = df_clean[~outlier_mask]
            
            outlier_report[col] = {
                'method': method,
                'threshold': threshold,
                'outliers_removed': int(outliers_found),
                'outlier_percentage': round((outliers_found / original_count) * 100, 2),
                'lower_bound': round(float(lower_bound), 6),
                'upper_bound': round(float(upper_bound), 6)
            }
            
            total_removed += outliers_found
        
        self.log_transformation('remove_outliers', {
            'method': method,
            'threshold': threshold,
            'columns_processed': columns,
            'total_outliers_removed': int(total_removed)
        })
        
        return df_clean, outlier_report
    
    def normalize_scale_data(self, 
                            df: pd.DataFrame, 
                            columns: Optional[List[str]] = None,
                            method: str = 'standard',
                            scaler_name: str = 'default') -> pd.DataFrame:
        """
        Normalize or scale numeric data.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (None for all numeric)
            method: 'standard', 'minmax', 'robust'
            scaler_name: Name to store scaler for later use
            
        Returns:
            DataFrame with scaled columns
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_columns:
            return df_scaled
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
        
        # Store scaler for potential inverse transformation
        self.scalers[scaler_name] = {'scaler': scaler, 'columns': numeric_columns, 'method': method}
        
        self.log_transformation('normalize_scale_data', {
            'method': method,
            'columns_processed': numeric_columns,
            'scaler_name': scaler_name
        })
        
        return df_scaled
    
    def encode_categorical(self, 
                          df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          method: str = 'onehot',
                          drop_first: bool = False,
                          encoder_name: str = 'default') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (None for all categorical)
            method: 'onehot', 'label', 'ordinal'
            drop_first: Drop first dummy variable (for onehot)
            encoder_name: Name to store encoder
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        categorical_columns = [col for col in columns if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        
        if not categorical_columns:
            return df_encoded
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=drop_first, prefix=categorical_columns)
            
        elif method == 'label':
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
            
            self.encoders[encoder_name] = {'encoders': label_encoders, 'method': method, 'columns': categorical_columns}
        
        elif method == 'ordinal':
            # For ordinal encoding, assume order by frequency (most common = 0)
            ordinal_encoders = {}
            for col in categorical_columns:
                value_counts = df_encoded[col].value_counts()
                ordinal_map = {val: idx for idx, val in enumerate(value_counts.index)}
                df_encoded[col] = df_encoded[col].map(ordinal_map)
                ordinal_encoders[col] = ordinal_map
            
            self.encoders[encoder_name] = {'encoders': ordinal_encoders, 'method': method, 'columns': categorical_columns}
        
        self.log_transformation('encode_categorical', {
            'method': method,
            'columns_processed': categorical_columns,
            'drop_first': drop_first,
            'encoder_name': encoder_name
        })
        
        return df_encoded
    
    def clean_text_columns(self, 
                          df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          operations: List[str] = ['strip', 'lower', 'remove_special']) -> pd.DataFrame:
        """
        Clean text columns.
        
        Args:
            df: Input DataFrame
            columns: Text columns to clean (None for all object columns)
            operations: List of operations ('strip', 'lower', 'upper', 'remove_special', 'remove_numbers')
            
        Returns:
            DataFrame with cleaned text columns
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = [col for col in df.columns if df[col].dtype == 'object']
        
        text_columns = [col for col in columns if col in df.columns and df[col].dtype == 'object']
        
        for col in text_columns:
            if 'strip' in operations:
                df_clean[col] = df_clean[col].astype(str).str.strip()
            
            if 'lower' in operations:
                df_clean[col] = df_clean[col].astype(str).str.lower()
            
            if 'upper' in operations:
                df_clean[col] = df_clean[col].astype(str).str.upper()
            
            if 'remove_special' in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            
            if 'remove_numbers' in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'\d+', '', regex=True)
        
        self.log_transformation('clean_text_columns', {
            'columns_processed': text_columns,
            'operations': operations
        })
        
        return df_clean
    
    def convert_data_types(self, 
                          df: pd.DataFrame, 
                          type_conversions: Dict[str, str]) -> pd.DataFrame:
        """
        Convert column data types.
        
        Args:
            df: Input DataFrame
            type_conversions: Dictionary mapping column names to target types
            
        Returns:
            DataFrame with converted types
        """
        df_converted = df.copy()
        successful_conversions = {}
        failed_conversions = {}
        
        for col, target_type in type_conversions.items():
            if col not in df_converted.columns:
                failed_conversions[col] = f"Column not found"
                continue
            
            try:
                if target_type.lower() in ['int', 'int64', 'integer']:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
                elif target_type.lower() in ['float', 'float64']:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                elif target_type.lower() in ['str', 'string', 'object']:
                    df_converted[col] = df_converted[col].astype(str)
                elif target_type.lower() in ['datetime', 'datetime64']:
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                elif target_type.lower() in ['bool', 'boolean']:
                    df_converted[col] = df_converted[col].astype('boolean')
                elif target_type.lower() == 'category':
                    df_converted[col] = df_converted[col].astype('category')
                else:
                    df_converted[col] = df_converted[col].astype(target_type)
                
                successful_conversions[col] = target_type
                
            except Exception as e:
                failed_conversions[col] = str(e)
        
        self.log_transformation('convert_data_types', {
            'successful_conversions': successful_conversions,
            'failed_conversions': failed_conversions
        })
        
        return df_converted
    
    def create_derived_features(self, 
                               df: pd.DataFrame, 
                               feature_definitions: Dict[str, str]) -> pd.DataFrame:
        """
        Create derived features using pandas operations.
        
        Args:
            df: Input DataFrame
            feature_definitions: Dictionary mapping new column names to pandas expressions
            
        Returns:
            DataFrame with new derived features
        """
        df_derived = df.copy()
        created_features = []
        failed_features = {}
        
        for new_col, expression in feature_definitions.items():
            try:
                # Use eval with the dataframe as context (be careful with user input in production)
                df_derived[new_col] = df_derived.eval(expression)
                created_features.append(new_col)
            except Exception as e:
                failed_features[new_col] = str(e)
        
        self.log_transformation('create_derived_features', {
            'created_features': created_features,
            'failed_features': failed_features,
            'feature_definitions': feature_definitions
        })
        
        return df_derived
    
    def remove_duplicates(self, 
                         df: pd.DataFrame, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates (None for all)
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        df_clean = df.copy()
        original_count = len(df_clean)
        
        df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
        
        duplicates_removed = original_count - len(df_clean)
        
        self.log_transformation('remove_duplicates', {
            'subset_columns': subset,
            'keep': keep,
            'duplicates_removed': duplicates_removed,
            'original_count': original_count,
            'final_count': len(df_clean)
        })
        
        return df_clean
    
    def filter_rows(self, 
                   df: pd.DataFrame, 
                   conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter rows based on conditions.
        
        Args:
            df: Input DataFrame
            conditions: Dictionary with filter conditions
                       e.g., {'age': {'>=': 18, '<': 65}, 'status': {'==': 'active'}}
            
        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        applied_filters = []
        
        for col, condition_dict in conditions.items():
            if col not in df_filtered.columns:
                continue
            
            for operator, value in condition_dict.items():
                original_count = len(df_filtered)
                
                if operator == '==':
                    df_filtered = df_filtered[df_filtered[col] == value]
                elif operator == '!=':
                    df_filtered = df_filtered[df_filtered[col] != value]
                elif operator == '>':
                    df_filtered = df_filtered[df_filtered[col] > value]
                elif operator == '>=':
                    df_filtered = df_filtered[df_filtered[col] >= value]
                elif operator == '<':
                    df_filtered = df_filtered[df_filtered[col] < value]
                elif operator == '<=':
                    df_filtered = df_filtered[df_filtered[col] <= value]
                elif operator == 'in':
                    df_filtered = df_filtered[df_filtered[col].isin(value)]
                elif operator == 'not_in':
                    df_filtered = df_filtered[~df_filtered[col].isin(value)]
                elif operator == 'contains':
                    df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(str(value), na=False)]
                
                rows_removed = original_count - len(df_filtered)
                applied_filters.append({
                    'column': col,
                    'operator': operator,
                    'value': value,
                    'rows_removed': rows_removed
                })
        
        self.log_transformation('filter_rows', {
            'applied_filters': applied_filters,
            'final_row_count': len(df_filtered)
        })
        
        return df_filtered
    

    # Example usage and testing
if __name__ == "__main__":
    # Initialize the tool
    transform_tool = DataTransformationTool()
    
    # Create sample data with various issues
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'Person_{i}' if i % 10 != 0 else None for i in range(1, 1001)],
        'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1000)],
        'salary': np.concatenate([np.random.normal(50000, 15000, 995), [200000, 300000, 10000, 400000, 500000]]),
        'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], 1000),
        'email': [f'person{i}@company.com  ' for i in range(1000)],  # Extra spaces
        'score': np.random.uniform(0, 100, 1000)
    })
    
    # Add some duplicates
    sample_data = pd.concat([sample_data, sample_data.tail(10)], ignore_index=True)
    
    print("=== Data Transformation Tool Demo ===")
    print(f"Original data: {len(sample_data)} rows, {len(sample_data.columns)} columns")
    
    # 1. Handle missing values
    print("\n1. Handling missing values...")
    cleaned_data = transform_tool.handle_missing_values(sample_data, strategy='mean')
    print(f"After handling missing: {cleaned_data.isna().sum().sum()} total missing values")
    
    # 2. Remove outliers
    print("\n2. Removing outliers...")
    no_outliers, outlier_report = transform_tool.remove_outliers(cleaned_data, columns=['salary'])
    print(f"Outliers removed: {outlier_report['salary']['outliers_removed']}")
    
    # 3. Clean text
    print("\n3. Cleaning text...")
    text_cleaned = transform_tool.clean_text_columns(no_outliers, columns=['email'])
    
    # 4. Remove duplicates
    print("\n4. Removing duplicates...")
    no_duplicates = transform_tool.remove_duplicates(text_cleaned)
    
    # 5. Encode categorical
    print("\n5. Encoding categorical variables...")
    encoded_data = transform_tool.encode_categorical(no_duplicates, columns=['department'])
    
    # 6. Scale numeric data
    print("\n6. Scaling numeric data...")
    scaled_data = transform_tool.normalize_scale_data(encoded_data, columns=['age', 'salary', 'score'])
    
    print(f"\nFinal data: {len(scaled_data)} rows, {len(scaled_data.columns)} columns")
    
    # Show transformation log
    print("\n=== Transformation Log ===")
    for i, log_entry in enumerate(transform_tool.get_transformation_log(), 1):
        print(f"{i}. {log_entry['operation']} at {log_entry['timestamp']}")