"""
DSPy File I/O Tool for Data Preprocessing Agent
Handles reading from spreadsheets and writing to various formats
"""

import pandas as pd
import os
from pathlib import Path
from typing import Union, Dict, List, Any, Optional
import json
import yaml


class FileIOTool:
    """
    A comprehensive file I/O tool for data preprocessing tasks.
    Supports reading CSV/Excel files and writing to multiple formats.
    """
    
    def __init__(self):
        self.supported_read_formats = ['.csv', '.xlsx', '.xls']
        self.supported_write_formats = ['.csv', '.xlsx', '.md', '.txt', '.json', '.yaml']
    
    def read_spreadsheet(self, 
                        file_path: str, 
                        sheet_name: Optional[str] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Read data from CSV or Excel files.
        
        Args:
            file_path: Path to the input file
            sheet_name: Sheet name for Excel files (None for first sheet)
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            pandas.DataFrame: The loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_read_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. "
                           f"Supported formats: {self.supported_read_formats}")
        
        try:
            if file_ext == '.csv':
                # Common CSV reading parameters
                default_csv_params = {
                    'encoding': 'utf-8',
                    'skipinitialspace': True,
                    'na_values': ['', 'N/A', 'NA', 'null', 'NULL']
                }
                default_csv_params.update(kwargs)
                df = pd.read_csv(file_path, **default_csv_params)
                
            elif file_ext in ['.xlsx', '.xls']:
                # Common Excel reading parameters
                default_excel_params = {
                    'sheet_name': sheet_name or 0,
                    'na_values': ['', 'N/A', 'NA', 'null', 'NULL']
                }
                default_excel_params.update(kwargs)
                df = pd.read_excel(file_path, **default_excel_params)
            
            print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
            return df
            
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def write_spreadsheet(self, 
                         df: pd.DataFrame, 
                         file_path: str, 
                         **kwargs) -> bool:
        """
        Write DataFrame to CSV or Excel format.
        
        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional parameters for pandas write functions
            
        Returns:
            bool: True if successful
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_ext == '.csv':
                default_csv_params = {
                    'index': False,
                    'encoding': 'utf-8'
                }
                default_csv_params.update(kwargs)
                df.to_csv(file_path, **default_csv_params)
                
            elif file_ext in ['.xlsx', '.xls']:
                default_excel_params = {
                    'index': False,
                    'engine': 'openpyxl' if file_ext == '.xlsx' else 'xlwt'
                }
                default_excel_params.update(kwargs)
                df.to_excel(file_path, **default_excel_params)
            
            else:
                raise ValueError(f"Unsupported spreadsheet format: {file_ext}")
            
            print(f"Successfully wrote {len(df)} rows to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing file {file_path}: {str(e)}")
            return False
    
    def write_markdown(self, 
                      content: Union[str, pd.DataFrame, Dict], 
                      file_path: str,
                      title: Optional[str] = None,
                      include_metadata: bool = True) -> bool:
        """
        Write content to markdown format.
        
        Args:
            content: Content to write (string, DataFrame, or dict)
            file_path: Output file path
            title: Optional title for the markdown document
            include_metadata: Whether to include metadata header
            
        Returns:
            bool: True if successful
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write metadata header if requested
                if include_metadata:
                    f.write("---\n")
                    f.write(f"title: {title or file_path.stem}\n")
                    f.write(f"generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("---\n\n")
                
                # Write title if provided
                if title:
                    f.write(f"# {title}\n\n")
                
                # Handle different content types
                if isinstance(content, pd.DataFrame):
                    f.write("## Data Overview\n\n")
                    f.write(f"- **Rows**: {len(content)}\n")
                    f.write(f"- **Columns**: {len(content.columns)}\n\n")
                    f.write("## Data\n\n")
                    f.write(content.to_markdown(index=False))
                    f.write("\n\n## Summary Statistics\n\n")
                    f.write(content.describe().to_markdown())
                    
                elif isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"## {key.replace('_', ' ').title()}\n\n")
                        if isinstance(value, pd.DataFrame):
                            f.write(value.to_markdown(index=False))
                        elif isinstance(value, list):
                            for item in value:
                                f.write(f"- {item}\n")
                        else:
                            f.write(f"{value}\n")
                        f.write("\n\n")
                        
                else:  # String content
                    f.write(str(content))
            
            print(f"Successfully wrote markdown to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing markdown file {file_path}: {str(e)}")
            return False
    
    def write_json(self, 
                   data: Union[Dict, List], 
                   file_path: str, 
                   indent: int = 2) -> bool:
        """
        Write data to JSON format.
        
        Args:
            data: Data to write (dict or list)
            file_path: Output file path
            indent: JSON indentation
            
        Returns:
            bool: True if successful
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            print(f"Successfully wrote JSON to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing JSON file {file_path}: {str(e)}")
            return False
    
    def write_text(self, 
                   content: str, 
                   file_path: str, 
                   encoding: str = 'utf-8') -> bool:
        """
        Write plain text content to file.
        
        Args:
            content: Text content to write
            file_path: Output file path
            encoding: File encoding
            
        Returns:
            bool: True if successful
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            print(f"Successfully wrote text to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing text file {file_path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        info = {
            "name": file_path.name,
            "path": str(file_path.absolute()),
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "extension": file_path.suffix,
            "modified": pd.Timestamp.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # If it's a spreadsheet, get additional info
        if file_path.suffix.lower() in self.supported_read_formats:
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path, nrows=0)  # Just get headers
                    info["columns"] = len(df.columns)
                    info["column_names"] = list(df.columns)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, nrows=0)  # Just get headers
                    info["columns"] = len(df.columns)
                    info["column_names"] = list(df.columns)
            except:
                info["read_error"] = "Could not read file headers"
        
        return info
    
    def list_files(self, 
                   directory: str, 
                   extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List files in a directory with optional extension filtering.
        
        Args:
            directory: Directory path
            extensions: List of extensions to filter (e.g., ['.csv', '.xlsx'])
            
        Returns:
            List of file information dictionaries
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        files = []
        for file_path in directory.iterdir():
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in extensions:
                    files.append(self.get_file_info(str(file_path)))
        
        return sorted(files, key=lambda x: x.get('name', ''))

# Example usage and testing
if __name__ == "__main__":
    # Initialize the tool
    io_tool = FileIOTool()
    
    # Example: Create sample data
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'London', 'Tokyo', 'Paris'],
        'Score': [85.5, 92.0, 78.5, 88.0]
    })
    
    # Example usage:
    print("=== File I/O Tool Demo ===")
    
    # Write to different formats
    io_tool.write_spreadsheet(sample_data, 'output/sample.csv')
    io_tool.write_spreadsheet(sample_data, 'output/sample.xlsx')
    
    # Write markdown report
    io_tool.write_markdown(sample_data, 'output/data_report.md', 
                          title='Sample Data Analysis')
    
    # Write JSON
    io_tool.write_json(sample_data.to_dict('records'), 'output/sample.json')
    
    # Get file info
    print("\n=== File Information ===")
    for file in ['output/sample.csv', 'output/sample.xlsx']:
        if Path(file).exists():
            info = io_tool.get_file_info(file)
            print(f"{file}: {info}")