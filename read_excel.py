
import pandas as pd

def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path, skiprows=3)
        print(df.head().to_markdown(index=False))
        print(df.info())
    except Exception as e:
        print(f"Error reading Excel file: {e}")

if __name__ == "__main__":
    read_excel_file("/CLEANEDDATA.xlsx")


