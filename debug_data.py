import pandas as pd
import os

def examine_excel_file(file_path):
    """Examine the structure of the Excel file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        # Read the Excel file without header first to see raw data
        print(f"Reading file: {file_path}")
        
        # Check all sheets
        excel_file = pd.ExcelFile(file_path)
        print(f"Available sheets: {excel_file.sheet_names}")
        
        # Read without header to see raw data
        df_raw = pd.read_excel(file_path, header=None)
        print(f"Raw data shape: {df_raw.shape}")
        print("\nFirst 15 rows of raw data:")
        print(df_raw.head(15))
        
        # Find the header row (look for row with 'YEAR')
        header_row = None
        for i in range(min(10, len(df_raw))):
            if 'YEAR' in df_raw.iloc[i].values:
                header_row = i
                print(f"Found header at row: {i}")
                break
        
        if header_row is None:
            print("Could not find header row with 'YEAR'")
            return
        
        # Read with correct header row
        df = pd.read_excel(file_path, header=header_row)
        print(f"\nData shape with header: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10))
        
        print("\nColumn dtypes:")
        print(df.dtypes)
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # Check unique values in key columns
        if 'YEAR' in df.columns:
            print("\nUnique years:", sorted(df['YEAR'].unique()))
        if 'SEX' in df.columns:
            print("Unique sexes:", df['SEX'].unique())
        if 'AGE GROUP' in df.columns:
            print("Unique age groups:", df['AGE GROUP'].unique())
        
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = "./data/CLEANEDDATA.xlsx"
    examine_excel_file(file_path)