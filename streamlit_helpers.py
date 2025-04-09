"""
Simple helper functions for Streamlit compatibility
"""
import pandas as pd

def prepare_df_for_streamlit(df):
    """
    Prepare a DataFrame for display in Streamlit by ensuring all columns
    have PyArrow-compatible data types
    
    Args:
        df (pandas.DataFrame): DataFrame to prepare
        
    Returns:
        pandas.DataFrame: DataFrame with PyArrow-compatible data types
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert all columns to strings
    for col in df.columns:
        df[col] = df[col].astype(str)
        
    # Replace 'None' and 'nan' strings with empty strings
    df = df.replace({'None': '', 'nan': '', 'NaN': ''})
    
    return df
