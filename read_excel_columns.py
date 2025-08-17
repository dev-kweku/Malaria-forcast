
import pandas as pd

try:
    df = pd.read_excel("/home/ubuntu/upload/CLEANEDDATA.xlsx")
    print("Columns in CLEANEDDATA.xlsx:")
    for col in df.columns:
        print(col)
except Exception as e:
    print(f"Error reading Excel file: {e}")


