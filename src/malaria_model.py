import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class MalariaModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['YEAR', 'SEX_encoded', 'AGE_encoded']
        self.sex_mapping = {'Male': 0, 'Female': 1}
        self.age_mapping = {
            '<28days': 0, '1-11mths': 1, '1-4': 2, '5-9': 3, '10-14': 4,
            '15-17': 5, '18-19': 6, '20-34': 7, '35-49': 8, '50-59': 9,
            '60-69': 10, '70+': 11
        }
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the Excel data with proper header handling"""
        try:
            # First read without header to find the correct header row
            df_raw = pd.read_excel(file_path, header=None)
            
            # Find the header row (look for row with 'YEAR')
            header_row = None
            for i in range(min(10, len(df_raw))):
                row_values = [str(x).strip() for x in df_raw.iloc[i].values if pd.notna(x)]
                if 'YEAR' in row_values:
                    header_row = i
                    print(f"Found header at row: {i}")
                    break
            
            if header_row is None:
                print("Could not find header row with 'YEAR'")
                return None
            
            # Read with correct header row
            df = pd.read_excel(file_path, header=header_row)
            print(f"Data shape after reading with header: {df.shape}")
            
            # Clean column names (remove extra spaces)
            df.columns = [str(col).strip() for col in df.columns]
            print(f"Cleaned columns: {list(df.columns)}")
            
            # Check if required columns exist
            required_columns = ['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Error: Missing columns: {missing_columns}")
                print("Available columns:", list(df.columns))
                return None
            
            # Filter data - keep only rows with Male/Female and valid age groups
            valid_sexes = ['Male', 'Female']
            df = df[df['SEX'].isin(valid_sexes)]
            
            valid_age_groups = list(self.age_mapping.keys())
            df = df[df['AGE GROUP'].isin(valid_age_groups)]
            
            print(f"Data shape after filtering: {df.shape}")
            
            # Convert numeric columns to appropriate types
            numeric_columns = ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing values in key columns
            df = df.dropna(subset=['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED TESTED', 'POSITIVE'])
            
            # Encode categorical variables
            df['SEX_encoded'] = df['SEX'].map(self.sex_mapping)
            df['AGE_encoded'] = df['AGE GROUP'].map(self.age_mapping)
            
            # Calculate positivity rate
            df['POSITIVITY_RATE'] = df['POSITIVE'] / df['SUSPECTED TESTED']
            
            # Remove invalid positivity rates
            df = df[(df['POSITIVITY_RATE'] >= 0) & (df['POSITIVITY_RATE'] <= 1)]
            
            print(f"Final data shape: {df.shape}")
            print(f"Years in data: {sorted(df['YEAR'].unique())}")
            print(f"Sex values: {df['SEX'].unique()}")
            print(f"Age groups: {df['AGE GROUP'].unique()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_training_data(self, df):
        """Prepare features and target for training"""
        X = df[self.features]
        y = df['POSITIVITY_RATE']
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return self.model
    
    def predict_future(self, years, sex, age_group):
        """Predict for future years"""
        predictions = {}
        
        for year in years:
            # Encode inputs
            sex_encoded = self.sex_mapping.get(sex, 0)
            age_encoded = self.age_mapping.get(age_group, 0)
            
            # Make prediction
            X_pred = np.array([[year, sex_encoded, age_encoded]])
            positivity_rate = self.model.predict(X_pred)[0]
            
            predictions[year] = max(0, min(positivity_rate, 1))  # Ensure between 0 and 1
        
        return predictions
    
    def save_model(self, file_path):
        """Save the trained model"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(file_path):
        """Load a saved model"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)