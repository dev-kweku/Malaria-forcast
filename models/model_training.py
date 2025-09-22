# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# class MalariaModel:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)
#         self.features = ['YEAR', 'SEX_encoded', 'AGE_encoded']
#         self.sex_mapping = {'Male': 0, 'Female': 1}
#         self.age_mapping = {
#             '<28days': 0, '1-11mths': 1, '1-4': 2, '5-9': 3, '10-14': 4,
#             '15-17': 5, '18-19': 6, '20-34': 7, '35-49': 8, '50-59': 9,
#             '60-69': 10, '70+': 11
#         }
    
#     def load_and_preprocess_data(self, file_path):
#         """Load and preprocess the Excel data with proper header handling"""
#         try:
#             # First read without header to find the correct header row
#             df_raw = pd.read_excel(file_path, header=None)
            
#             # Find the header row (look for row with 'YEAR')
#             header_row = None
#             for i in range(min(10, len(df_raw))):
#                 if 'YEAR' in df_raw.iloc[i].values:
#                     header_row = i
#                     print(f"Found header at row: {i}")
#                     break
            
#             if header_row is None:
#                 print("Could not find header row with 'YEAR'")
#                 return None
            
#             # Read with correct header row
#             df = pd.read_excel(file_path, header=header_row)
#             print(f"Data shape after reading with header: {df.shape}")
            
#             # Clean column names (remove extra spaces)
#             df.columns = [str(col).strip() for col in df.columns]
#             print(f"Cleaned columns: {list(df.columns)}")
            
#             # Check if required columns exist
#             required_columns = ['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
#             missing_columns = [col for col in required_columns if col not in df.columns]
            
#             if missing_columns:
#                 print(f"Error: Missing columns: {missing_columns}")
#                 print("Available columns:", list(df.columns))
#                 return None
            
#             # Filter data - keep only rows with Male/Female and valid age groups
#             valid_sexes = ['Male', 'Female']
#             df = df[df['SEX'].isin(valid_sexes)]
            
#             valid_age_groups = list(self.age_mapping.keys())
#             df = df[df['AGE GROUP'].isin(valid_age_groups)]
            
#             print(f"Data shape after filtering: {df.shape}")
            
#             # Convert numeric columns to appropriate types
#             numeric_columns = ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
#             for col in numeric_columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Remove rows with missing values in key columns
#             df = df.dropna(subset=['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED TESTED', 'POSITIVE'])
            
#             # Encode categorical variables
#             df['SEX_encoded'] = df['SEX'].map(self.sex_mapping)
#             df['AGE_encoded'] = df['AGE GROUP'].map(self.age_mapping)
            
#             # Calculate positivity rate
#             df['POSITIVITY_RATE'] = df['POSITIVE'] / df['SUSPECTED TESTED']
            
#             # Remove invalid positivity rates
#             df = df[(df['POSITIVITY_RATE'] >= 0) & (df['POSITIVITY_RATE'] <= 1)]
            
#             print(f"Final data shape: {df.shape}")
#             print(f"Years in data: {sorted(df['YEAR'].unique())}")
#             print(f"Sex values: {df['SEX'].unique()}")
#             print(f"Age groups: {df['AGE GROUP'].unique()}")
            
#             return df
            
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
    
#     def prepare_training_data(self, df):
#         """Prepare features and target for training"""
#         X = df[self.features]
#         y = df['POSITIVITY_RATE']
#         return X, y
    
#     def train(self, X, y):
#         """Train the model"""
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
        
#         self.model.fit(X_train, y_train)
        
#         # Evaluate model
#         y_pred = self.model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         print(f"Model Performance:")
#         print(f"MAE: {mae:.4f}")
#         print(f"R² Score: {r2:.4f}")
        
#         # Feature importance
#         feature_importance = pd.DataFrame({
#             'feature': self.features,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\nFeature Importance:")
#         print(feature_importance)
        
#         return self.model
    
#     def predict_future(self, years, sex, age_group):
#         """Predict for future years"""
#         predictions = {}
        
#         for year in years:
#             # Encode inputs
#             sex_encoded = self.sex_mapping.get(sex, 0)
#             age_encoded = self.age_mapping.get(age_group, 0)
            
#             # Make prediction
#             X_pred = np.array([[year, sex_encoded, age_encoded]])
#             positivity_rate = self.model.predict(X_pred)[0]
            
#             predictions[year] = max(0, min(positivity_rate, 1))  # Ensure between 0 and 1
        
#         return predictions
    
#     def save_model(self, file_path):
#         """Save the trained model"""
#         with open(file_path, 'wb') as f:
#             pickle.dump(self, f)
    
#     @staticmethod
#     def load_model(file_path):
#         """Load a saved model"""
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)

# # Alternative method: Manual column assignment
# def load_data_manually(file_path):
#     """Load data by manually assigning column names"""
#     try:
#         # Read without header
#         df = pd.read_excel(file_path, header=None)
        
#         # Find the row with column names
#         header_idx = None
#         for i in range(len(df)):
#             if df.iloc[i, 0] == 'YEAR':
#                 header_idx = i
#                 break
        
#         if header_idx is None:
#             print("Could not find header row")
#             return None
        
#         # Get column names from that row
#         column_names = df.iloc[header_idx].tolist()
#         print(f"Column names found: {column_names}")
        
#         # Read data starting from next row
#         df = pd.read_excel(file_path, header=header_idx+1, names=column_names)
        
#         # Clean column names
#         df.columns = [str(col).strip() for col in df.columns]
        
#         return df
        
#     except Exception as e:
#         print(f"Error in manual loading: {e}")
#         return None

# # Train and save the model
# if __name__ == "__main__":
#     # Initialize and train model
#     malaria_model = MalariaModel()
    
#     # Try different loading methods
#     file_path = 'data/CLEANEDDATA.xlsx'
    
#     print("Method 1: Auto-detect header")
#     df = malaria_model.load_and_preprocess_data(file_path)
    
#     if df is None or len(df) == 0:
#         print("\nMethod 2: Manual column assignment")
#         df = load_data_manually(file_path)
#         if df is not None:
#             print(f"Manual load successful. Shape: {df.shape}")
#             print(f"Columns: {list(df.columns)}")
            
#             # Continue with manual processing
#             valid_sexes = ['Male', 'Female']
#             df = df[df['SEX'].isin(valid_sexes)]
            
#             valid_age_groups = list(malaria_model.age_mapping.keys())
#             df = df[df['AGE GROUP'].isin(valid_age_groups)]
            
#             # Convert numeric columns
#             numeric_columns = ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
#             for col in numeric_columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             df = df.dropna(subset=numeric_columns)
            
#             # Encode and calculate positivity rate
#             df['SEX_encoded'] = df['SEX'].map(malaria_model.sex_mapping)
#             df['AGE_encoded'] = df['AGE GROUP'].map(malaria_model.age_mapping)
#             df['POSITIVITY_RATE'] = df['POSITIVE'] / df['SUSPECTED TESTED']
#             df = df[(df['POSITIVITY_RATE'] >= 0) & (df['POSITIVITY_RATE'] <= 1)]
    
#     if df is None or len(df) == 0:
#         print("Could not load data. Please check the file structure.")
#     else:
#         X, y = malaria_model.prepare_training_data(df)
#         print(f"Training on {X.shape[0]} samples")
        
#         model = malaria_model.train(X, y)
        
#         # Save the model
#         malaria_model.save_model('/models/malaria_model.pkl')
#         print("Model saved successfully!")
        
#         # Test prediction
#         test_pred = malaria_model.predict_future([2025, 2026], 'Male', '20-34')
#         print(f"Test predictions for Male, 20-34: {test_pred}")




import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

class MalariaModel:
    def __init__(self):
        self.model = None
        self.features = ['YEAR', 'SEX_encoded', 'AGE_encoded']
        self.sex_mapping = {'Male': 0, 'Female': 1}
        self.age_mapping = {
            '<28days': 0, '1-11mths': 1, '1-4': 2, '5-9': 3, '10-14': 4,
            '15-17': 5, '18-19': 6, '20-34': 7, '35-49': 8, '50-59': 9,
            '60-69': 10, '70+': 11
        }
        self.feature_names = None
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the Excel data with proper header handling"""
        try:
            # First read without header to find the correct header row
            df_raw = pd.read_excel(file_path, header=None)
            
            # Find the header row (look for row with 'YEAR')
            header_row = None
            for i in range(min(10, len(df_raw))):
                if 'YEAR' in df_raw.iloc[i].values:
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
            
            # Convert numeric columns to appropriate types
            numeric_columns = ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing values in key columns
            df = df.dropna(subset=['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED TESTED', 'POSITIVE'])
            
            # Filter data - keep only rows with Male/Female and valid age groups
            valid_sexes = ['Male', 'Female']
            df = df[df['SEX'].isin(valid_sexes)]
            
            valid_age_groups = list(self.age_mapping.keys())
            df = df[df['AGE GROUP'].isin(valid_age_groups)]
            
            print(f"Data shape after filtering: {df.shape}")
            
            # Aggregate data by year, sex, and age group
            agg_df = df.groupby(['YEAR', 'SEX', 'AGE GROUP']).agg({
                'SUSPECTED TESTED': 'sum',
                'POSITIVE': 'sum'
            }).reset_index()
            
            print(f"Data shape after aggregation: {agg_df.shape}")
            
            # Calculate positivity rate
            agg_df['POSITIVITY_RATE'] = agg_df['POSITIVE'] / agg_df['SUSPECTED TESTED']
            
            # Remove invalid positivity rates
            agg_df = agg_df[(agg_df['POSITIVITY_RATE'] >= 0) & (agg_df['POSITIVITY_RATE'] <= 1)]
            
            # Encode categorical variables
            agg_df['SEX_encoded'] = agg_df['SEX'].map(self.sex_mapping)
            agg_df['AGE_encoded'] = agg_df['AGE GROUP'].map(self.age_mapping)
            
            print(f"Final data shape: {agg_df.shape}")
            print(f"Years in data: {sorted(agg_df['YEAR'].unique())}")
            print(f"Sex values: {agg_df['SEX'].unique()}")
            print(f"Age groups: {agg_df['AGE GROUP'].unique()}")
            
            return agg_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_training_data(self, df):
        """Prepare features and target for training"""
        X = df[self.features]
        y = df['POSITIVITY_RATE']
        self.feature_names = self.features
        return X, y
    
    def train(self, X, y):
        """Train the model with optimized parameters"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create a pipeline with polynomial features and Ridge regression
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', Ridge(alpha=0.1))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0],
            'poly__degree': [1, 2]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Model Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"Cross-validated R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        # Feature importance (for polynomial features)
        if hasattr(self.model.named_steps['model'], 'coef_'):
            feature_names = self.model.named_steps['poly'].get_feature_names_out(input_features=self.features)
            coefs = self.model.named_steps['model'].coef_
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefs)
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance.head(10))
        
        return self.model
    
    def predict_future(self, years, sex, age_group):
        """Predict for future years"""
        predictions = {}
        
        for year in years:
            # Encode inputs
            sex_encoded = self.sex_mapping.get(sex, 0)
            age_encoded = self.age_mapping.get(age_group, 0)
            
            # Create feature array
            X_pred = np.array([[year, sex_encoded, age_encoded]])
            
            # Make prediction
            positivity_rate = self.model.predict(X_pred)[0]
            
            predictions[year] = max(0, min(positivity_rate, 1))  # Ensure between 0 and 1
        
        return predictions
    
    def save_model(self, file_path):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'sex_mapping': self.sex_mapping,
            'age_mapping': self.age_mapping,
            'feature_names': self.feature_names
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(file_path):
        """Load a saved model"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

# Train and save the model
if __name__ == "__main__":
    # Initialize and train model
    malaria_model = MalariaModel()
    
    # Load data
    file_path = 'data/CLEANEDDATA.xlsx'
    df = malaria_model.load_and_preprocess_data(file_path)
    
    if df is None or len(df) == 0:
        print("Could not load data. Please check the file structure.")
    else:
        X, y = malaria_model.prepare_training_data(df)
        print(f"Training on {X.shape[0]} samples")
        
        model = malaria_model.train(X, y)
        
        # Save the model
        model_path = 'models/malaria_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        malaria_model.save_model(model_path)
        print(f"Model saved successfully to {model_path}!")
        
        # Test prediction
        test_pred = malaria_model.predict_future([2025, 2026], 'Male', '20-34')
        print(f"Test predictions for Male, 20-34: {test_pred}")
        
        # Test with different demographics
        test_pred2 = malaria_model.predict_future([2025, 2026], 'Female', '1-11mths')
        print(f"Test predictions for Female, 1-11mths: {test_pred2}")






# import pandas as pd
# import os
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.pipeline import Pipeline
# import xgboost as xgb
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# class MalariaModel:
#     def __init__(self):
#         self.model = None
#         self.features = ['YEAR', 'SEX_encoded', 'AGE_encoded']
#         self.sex_mapping = {'Male': 0, 'Female': 1}
#         self.age_mapping = {
#             '<28days': 0, '1-11mths': 1, '1-4': 2, '5-9': 3, '10-14': 4,
#             '15-17': 5, '18-19': 6, '20-34': 7, '35-49': 8, '50-59': 9,
#             '60-69': 10, '70+': 11
#         }
#         self.feature_names = None
#         self.scaler = StandardScaler()
#         self.poly = PolynomialFeatures(degree=2, include_bias=False)
    
#     def load_and_preprocess_data(self, file_path):
#         """Load and preprocess the Excel data with proper header handling"""
#         try:
#             # First read without header to find the correct header row
#             df_raw = pd.read_excel(file_path, header=None)
            
#             # Find the header row (look for row with 'YEAR')
#             header_row = None
#             for i in range(min(10, len(df_raw))):
#                 if 'YEAR' in df_raw.iloc[i].values:
#                     header_row = i
#                     print(f"Found header at row: {i}")
#                     break
            
#             if header_row is None:
#                 print("Could not find header row with 'YEAR'")
#                 return None
            
#             # Read with correct header row
#             df = pd.read_excel(file_path, header=header_row)
#             print(f"Data shape after reading with header: {df.shape}")
            
#             # Clean column names (remove extra spaces)
#             df.columns = [str(col).strip() for col in df.columns]
#             print(f"Cleaned columns: {list(df.columns)}")
            
#             # Check if required columns exist
#             required_columns = ['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
#             missing_columns = [col for col in required_columns if col not in df.columns]
            
#             if missing_columns:
#                 print(f"Error: Missing columns: {missing_columns}")
#                 print("Available columns:", list(df.columns))
#                 return None
            
#             # Convert numeric columns to appropriate types
#             numeric_columns = ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
#             for col in numeric_columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Remove rows with missing values in key columns
#             df = df.dropna(subset=['YEAR', 'SEX', 'AGE GROUP', 'SUSPECTED TESTED', 'POSITIVE'])
            
#             # Filter data - keep only rows with Male/Female and valid age groups
#             valid_sexes = ['Male', 'Female']
#             df = df[df['SEX'].isin(valid_sexes)]
            
#             valid_age_groups = list(self.age_mapping.keys())
#             df = df[df['AGE GROUP'].isin(valid_age_groups)]
            
#             print(f"Data shape after filtering: {df.shape}")
            
#             # Aggregate data by year, sex, and age group
#             agg_df = df.groupby(['YEAR', 'SEX', 'AGE GROUP']).agg({
#                 'SUSPECTED TESTED': 'sum',
#                 'POSITIVE': 'sum'
#             }).reset_index()
            
#             print(f"Data shape after aggregation: {agg_df.shape}")
            
#             # Calculate positivity rate
#             agg_df['POSITIVITY_RATE'] = agg_df['POSITIVE'] / agg_df['SUSPECTED TESTED']
            
#             # Remove invalid positivity rates
#             agg_df = agg_df[(agg_df['POSITIVITY_RATE'] >= 0) & (agg_df['POSITIVITY_RATE'] <= 1)]
            
#             # Encode categorical variables
#             agg_df['SEX_encoded'] = agg_df['SEX'].map(self.sex_mapping)
#             agg_df['AGE_encoded'] = agg_df['AGE GROUP'].map(self.age_mapping)
            
#             # Add time-based features
#             agg_df['YEARS_SINCE_START'] = agg_df['YEAR'] - agg_df['YEAR'].min()
            
#             print(f"Final data shape: {agg_df.shape}")
#             print(f"Years in data: {sorted(agg_df['YEAR'].unique())}")
#             print(f"Sex values: {agg_df['SEX'].unique()}")
#             print(f"Age groups: {agg_df['AGE GROUP'].unique()}")
            
#             return agg_df
            
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
    
#     def prepare_training_data(self, df):
#         """Prepare features and target for training"""
#         # Add interaction features
#         df['YEAR_AGE_INTERACTION'] = df['YEAR'] * df['AGE_encoded']
#         df['YEAR_SEX_INTERACTION'] = df['YEAR'] * df['SEX_encoded']
        
#         # Update features list
#         self.features = ['YEAR', 'SEX_encoded', 'AGE_encoded', 
#                          'YEARS_SINCE_START', 'YEAR_AGE_INTERACTION', 'YEAR_SEX_INTERACTION']
        
#         X = df[self.features]
#         y = df['POSITIVITY_RATE']
#         self.feature_names = self.features
#         return X, y
    
#     def train(self, X, y):
#         """Train the model with optimized parameters"""
#         # Use TimeSeriesSplit for time-series data
#         tscv = TimeSeriesSplit(n_splits=3)
        
#         # Define models to try
#         models = {
#             'ridge': Pipeline([
#                 ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#                 ('scaler', StandardScaler()),
#                 ('model', Ridge())
#             ]),
#             'random_forest': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('model', RandomForestRegressor(random_state=42))
#             ]),
#             'xgboost': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('model', xgb.XGBRegressor(random_state=42))
#             ])
#         }
        
#         # Define hyperparameter grids
#         param_grids = {
#             'ridge': {
#                 'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
#                 'poly__degree': [1, 2]
#             },
#             'random_forest': {
#                 'model__n_estimators': [50, 100, 200],
#                 'model__max_depth': [3, 5, 7],
#                 'model__min_samples_split': [2, 5],
#                 'model__min_samples_leaf': [1, 2]
#             },
#             'xgboost': {
#                 'model__n_estimators': [50, 100],
#                 'model__max_depth': [3, 5],
#                 'model__learning_rate': [0.01, 0.1],
#                 'model__subsample': [0.8, 1.0]
#             }
#         }
        
#         best_model = None
#         best_score = float('inf')
#         best_model_name = None
        
#         # Try each model
#         for model_name, model in models.items():
#             print(f"\nEvaluating {model_name}...")
            
#             # Grid search with time series cross-validation
#             grid_search = GridSearchCV(
#                 model, param_grids[model_name], 
#                 cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
#             )
            
#             grid_search.fit(X, y)
            
#             # Get best score
#             cv_score = -grid_search.best_score_
#             print(f"Best MAE for {model_name}: {cv_score:.4f}")
#             print(f"Best parameters: {grid_search.best_params_}")
            
#             # Update best model if current model is better
#             if cv_score < best_score:
#                 best_score = cv_score
#                 best_model = grid_search.best_estimator_
#                 best_model_name = model_name
        
#         # Set the best model
#         self.model = best_model
#         print(f"\nSelected model: {best_model_name} with MAE: {best_score:.4f}")
        
#         # Evaluate on full dataset with cross-validation
#         cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
#         cv_mae = -cv_scores.mean()
#         cv_std = cv_scores.std()
        
#         # Calculate R²
#         y_pred = self.model.predict(X)
#         r2 = r2_score(y, y_pred)
#         mse = mean_squared_error(y, y_pred)
#         rmse = np.sqrt(mse)
        
#         print(f"\nModel Performance:")
#         print(f"Cross-validated MAE: {cv_mae:.4f} (±{cv_std:.4f})")
#         print(f"R² Score: {r2:.4f}")
#         print(f"RMSE: {rmse:.4f}")
        
#         # Feature importance for tree-based models
#         if best_model_name in ['random_forest', 'xgboost']:
#             if hasattr(self.model.named_steps['model'], 'feature_importances_'):
#                 importances = self.model.named_steps['model'].feature_importances_
#                 feature_importance = pd.DataFrame({
#                     'feature': self.feature_names,
#                     'importance': importances
#                 }).sort_values('importance', ascending=False)
                
#                 print("\nFeature Importance:")
#                 print(feature_importance)
        
#         return self.model
    
#     def predict_future(self, years, sex, age_group):
#         """Predict for future years"""
#         predictions = {}
        
#         for year in years:
#             # Encode inputs
#             sex_encoded = self.sex_mapping.get(sex, 0)
#             age_encoded = self.age_mapping.get(age_group, 0)
            
#             # Create feature array with all features
#             years_since_start = year - 2019  # Assuming 2019 is the first year in data
#             year_age_interaction = year * age_encoded
#             year_sex_interaction = year * sex_encoded
            
#             X_pred = np.array([[
#                 year, sex_encoded, age_encoded,
#                 years_since_start, year_age_interaction, year_sex_interaction
#             ]])
            
#             # Make prediction
#             positivity_rate = self.model.predict(X_pred)[0]
            
#             # Ensure between 0 and 1
#             predictions[year] = max(0, min(positivity_rate, 1))
        
#         return predictions
    
#     def save_model(self, file_path):
#         """Save the trained model"""
#         model_data = {
#             'model': self.model,
#             'sex_mapping': self.sex_mapping,
#             'age_mapping': self.age_mapping,
#             'feature_names': self.feature_names
#         }
#         with open(file_path, 'wb') as f:
#             pickle.dump(model_data, f)
    
#     @staticmethod
#     def load_model(file_path):
#         """Load a saved model"""
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)

# # Train and save the model
# if __name__ == "__main__":
#     # Initialize and train model
#     malaria_model = MalariaModel()
    
#     # Load data
#     file_path = 'data/CLEANEDDATA.xlsx'
#     df = malaria_model.load_and_preprocess_data(file_path)
    
#     if df is None or len(df) == 0:
#         print("Could not load data. Please check the file structure.")
#     else:
#         X, y = malaria_model.prepare_training_data(df)
#         print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        
#         model = malaria_model.train(X, y)
        
#         # Save the model
#         model_path = 'models/malaria_model.pkl'
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         malaria_model.save_model(model_path)
#         print(f"Model saved successfully to {model_path}!")
        
#         # Test prediction
#         test_pred = malaria_model.predict_future([2025, 2026], 'Male', '20-34')
#         print(f"\nTest predictions for Male, 20-34: {test_pred}")
        
#         # Test with different demographics
#         test_pred2 = malaria_model.predict_future([2025, 2026], 'Female', '1-11mths')
#         print(f"Test predictions for Female, 1-11mths: {test_pred2}")