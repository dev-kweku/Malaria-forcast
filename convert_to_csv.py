import pandas as pd
import os

def convert_excel_to_csv(excel_path, csv_path):
    """Convert Excel file to CSV with proper formatting"""
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Clean column names
        df.columns = [str(col).strip().title() for col in df.columns]
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        print(f"Converted {excel_path} to {csv_path}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    convert_excel_to_csv('./data/CLEANEDDATA.xlsx', './data/cleaned_data.csv')