
import pandas as pd

try:
    df = pd.read_excel("/CLEANEDDATA.xlsx")
    print("First 5 rows of CLEANEDDATA.xlsx:")
    print(df.head().to_markdown(index=False))
    print("\nColumn names and their data types:")
    print(df.info())
except Exception as e:
    print(f"Error reading Excel file: {e}")


