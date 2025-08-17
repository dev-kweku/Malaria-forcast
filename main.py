import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import optuna
# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malaria_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MalariaPredictor:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'Total Cases'
        self.feature_means = {}
        self.best_model_type = None
        self.scaler_type = 'standard'
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess malaria and environmental data"""
        try:
            # Use data_dir parameter for file paths
            malaria_path = self.data_dir / 'CLEANEDDATA.xlsx'
            env_path = self.data_dir / 'accra_environmental_data.csv'
            
            # Check if files exist
            if not malaria_path.exists():
                raise FileNotFoundError(f"Malaria data file not found: {malaria_path}")
            if not env_path.exists():
                raise FileNotFoundError(f"Environmental data file not found: {env_path}")
            
            # Load malaria data
            malaria_df = pd.read_excel(malaria_path, skiprows=3)
            malaria_df.columns = ["YEAR", "SEX", "AGE GROUP", "SUSPECTED", 
                                "SUSPECTED TESTED", "POSITIVE"]
            
            # Aggregate yearly cases
            malaria_yearly = malaria_df.groupby("YEAR")["POSITIVE"].sum().reset_index()
            malaria_yearly.columns = ["Year", self.target_column]
            
            # Load environmental data
            environmental_df = pd.read_csv(env_path)
            environmental_df["Date"] = pd.to_datetime(environmental_df["Date"])
            environmental_df["Year"] = environmental_df["Date"].dt.year
            
            # Validate environmental data has required columns
            required_env_columns = ["Temperature_2m", "Relative_Humidity_2m", "Precipitation"]
            missing_columns = [col for col in required_env_columns if col not in environmental_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in environmental data: {missing_columns}")
            
            # Calculate yearly environmental metrics
            environmental_yearly = environmental_df.groupby("Year").agg({
                "Temperature_2m": ["mean", "std", "max", "min"],
                "Relative_Humidity_2m": ["mean", "std"],
                "Precipitation": ["sum", "mean", "max"]
            }).reset_index()
            
            # Flatten multi-index columns
            environmental_yearly.columns = [
                'Year', 'T2M_mean', 'T2M_std', 'T2M_max', 'T2M_min',
                'RH2M_mean', 'RH2M_std', 'PRECTOTCORR_sum', 
                'PRECTOTCORR_mean', 'PRECTOTCORR_max'
            ]
            
            return malaria_yearly, environmental_yearly
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def create_features(self, malaria_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
        """Merge datasets and create features"""
        try:
            # Validate input data
            if "Year" not in malaria_df.columns or self.target_column not in malaria_df.columns:
                raise ValueError("Malaria data must contain 'Year' and target columns")
            if "Year" not in env_df.columns:
                raise ValueError("Environmental data must contain 'Year' column")
            
            # Merge datasets
            df = pd.merge(malaria_df, env_df, on="Year", how="inner")
            
            # Validate we have sufficient data
            if len(df) < 5:
                logger.warning(f"Small dataset with only {len(df)} years. This may impact model performance.")
                
            # Create lag features for target and environmental variables
            lag_periods = [1, 2]  # Reduced from 3 to preserve more data
            
            # Target variable lags
            for lag in lag_periods:
                df[f"{self.target_column}_Lag_{lag}"] = df[self.target_column].shift(lag)
            
            # Environmental variable lags
            env_vars = ['T2M_mean', 'RH2M_mean', 'PRECTOTCORR_sum']
            for var in env_vars:
                if var not in df.columns:
                    logger.warning(f"Environmental variable {var} not found in data. Skipping.")
                    continue
                for lag in lag_periods:
                    df[f"{var}_Lag_{lag}"] = df[var].shift(lag)
            
            # Add rolling statistics - reduced window size
            for window in [2]:  # Only use window of 2 to preserve data
                df[f"{self.target_column}_RollingMean_{window}"] = (
                    df[self.target_column].rolling(window=window).mean().shift(1))
                df[f"{self.target_column}_RollingStd_{window}"] = (
                    df[self.target_column].rolling(window=window).std().shift(1))
            
            # Add year-over-year differences
            df[f"{self.target_column}_YoY"] = (
                df[self.target_column] / df[f"{self.target_column}_Lag_1"] - 1)
            
            # Add interaction features
            df['Temp_Humidity_Interaction'] = df['T2M_mean'] * df['RH2M_mean']
            df['Temp_Precip_Interaction'] = df['T2M_mean'] * df['PRECTOTCORR_sum']
            df['Humidity_Precip_Interaction'] = df['RH2M_mean'] * df['PRECTOTCORR_sum']
            
            # Add trend feature
            df['Year_Trend'] = range(1, len(df) + 1)
            
            # Drop rows with missing values from lag features
            df.dropna(inplace=True)
            
            # Save historical data for dashboard
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            df.to_csv(output_dir / 'historical_data.csv', index=False)
            
            logger.info(f"Created features: {df.columns.tolist()}")
            logger.info(f"Final dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise
            
    def train_model(self, df: pd.DataFrame, test_year: int = 2024) -> Dict:
        """Train and evaluate the model"""
        try:
            # Validate input data
            if self.target_column not in df.columns or "Year" not in df.columns:
                raise ValueError("Data must contain target column and 'Year' column")
            
            # Check if test_year exists in data
            if test_year not in df["Year"].values:
                raise ValueError(f"Test year {test_year} not found in data")
            
            # Split data
            X = df.drop([self.target_column, "Year"], axis=1)
            y = df[self.target_column]
            self.feature_columns = X.columns.tolist()
            
            # Store feature means for forecasting
            self.feature_means = X.mean().to_dict()
            
            # Time-based split - use multiple years for testing if possible
            test_years = [test_year]
            if test_year - 1 in df["Year"].values and len(df) > 6:
                test_years.append(test_year - 1)
                logger.info(f"Using multiple test years: {test_years}")
            
            X_train = X[~df["Year"].isin(test_years)]
            y_train = y[~df["Year"].isin(test_years)]
            X_test = X[df["Year"].isin(test_years)]
            y_test = y[df["Year"].isin(test_years)]
            
            if len(X_test) == 0:
                raise ValueError(f"No data available for test years {test_years}")
            
            # Determine the best model type based on dataset size
            n_train_samples = len(X_train)
            
            if n_train_samples < 5:
                logger.warning(f"Very small dataset ({n_train_samples} samples). Using simple models.")
                model_types = ['linear', 'rf']
                cv_method = LeaveOneOut()
            elif n_train_samples < 10:
                logger.warning(f"Small dataset ({n_train_samples} samples). Using robust models.")
                model_types = ['rf', 'xgb']
                cv_method = TimeSeriesSplit(n_splits=min(3, n_train_samples - 1))
            else:
                logger.info(f"Medium dataset ({n_train_samples} samples). Using advanced models.")
                model_types = ['xgb', 'rf']
                cv_method = TimeSeriesSplit(n_splits=min(5, n_train_samples - 1))
            
            # Try different scalers
            scalers = {
                'standard': StandardScaler(),
                'robust': RobustScaler(),
                'none': None
            }
            
            best_score = float('inf')
            best_model = None
            best_scaler = None
            best_model_type = None
            
            for scaler_name, scaler in scalers.items():
                for model_type in model_types:
                    logger.info(f"Trying {model_type} model with {scaler_name} scaler")
                    
                    # Define model
                    if model_type == 'xgb':
                        def objective(trial):
                            params = {
                                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                                'max_depth': trial.suggest_int('max_depth', 2, 8),
                                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                                'gamma': trial.suggest_float('gamma', 0, 0.5),
                                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                            }
                            
                            model = XGBRegressor(objective='reg:squarederror', **params)
                            
                            # Create pipeline with scaler
                            if scaler:
                                pipeline = Pipeline([
                                    ('scaler', scaler),
                                    ('model', model)
                                ])
                            else:
                                pipeline = model
                            
                            # Evaluate with cross-validation
                            cv_scores = []
                            for train_idx, val_idx in cv_method.split(X_train):
                                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                                
                                pipeline.fit(X_cv_train, y_cv_train)
                                preds = pipeline.predict(X_cv_val)
                                cv_scores.append(mean_absolute_error(y_cv_val, preds))
                                
                            return np.mean(cv_scores)
                        
                        # Hyperparameter optimization
                        study = optuna.create_study(direction='minimize')
                        study.optimize(objective, n_trials=min(50, n_train_samples * 3))
                        
                        # Train final model with best params
                        best_params = study.best_params
                        model = XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=best_params['n_estimators'],
                            max_depth=best_params['max_depth'],
                            learning_rate=best_params['learning_rate'],
                            subsample=best_params['subsample'],
                            colsample_bytree=best_params['colsample_bytree'],
                            gamma=best_params['gamma'],
                            reg_alpha=best_params['reg_alpha'],
                            reg_lambda=best_params['reg_lambda'],
                            random_state=42
                        )
                    
                    elif model_type == 'rf':
                        def objective(trial):
                            params = {
                                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                                'max_depth': trial.suggest_int('max_depth', 2, 10),
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                                'max_features': trial.suggest_float('max_features', 0.6, 1.0)
                            }
                            
                            model = RandomForestRegressor(**params, random_state=42)
                            
                            # Create pipeline with scaler
                            if scaler:
                                pipeline = Pipeline([
                                    ('scaler', scaler),
                                    ('model', model)
                                ])
                            else:
                                pipeline = model
                            
                            # Evaluate with cross-validation
                            cv_scores = []
                            for train_idx, val_idx in cv_method.split(X_train):
                                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                                
                                pipeline.fit(X_cv_train, y_cv_train)
                                preds = pipeline.predict(X_cv_val)
                                cv_scores.append(mean_absolute_error(y_cv_val, preds))
                                
                            return np.mean(cv_scores)
                        
                        # Hyperparameter optimization
                        study = optuna.create_study(direction='minimize')
                        study.optimize(objective, n_trials=min(50, n_train_samples * 3))
                        
                        # Train final model with best params
                        best_params = study.best_params
                        model = RandomForestRegressor(
                            n_estimators=best_params['n_estimators'],
                            max_depth=best_params['max_depth'],
                            min_samples_split=best_params['min_samples_split'],
                            min_samples_leaf=best_params['min_samples_leaf'],
                            max_features=best_params['max_features'],
                            random_state=42
                        )
                    
                    elif model_type == 'linear':
                        model = LinearRegression()
                    
                    # Create pipeline with scaler
                    if scaler:
                        pipeline = Pipeline([
                            ('scaler', scaler),
                            ('model', model)
                        ])
                    else:
                        pipeline = model
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate on validation set (if we have enough data)
                    if n_train_samples > 3:
                        # Create a simple validation split
                        val_size = min(0.2, 1 / n_train_samples)  # Ensure at least 1 sample for validation
                        X_tr, X_val, y_tr, y_val = train_test_split(
                            X_train, y_train, test_size=val_size, shuffle=False
                        )
                        
                        pipeline.fit(X_tr, y_tr)
                        val_preds = pipeline.predict(X_val)
                        val_score = mean_absolute_error(y_val, val_preds)
                    else:
                        # Use training score as validation score
                        val_score = mean_absolute_error(y_train, pipeline.predict(X_train))
                    
                    # Update best model if this one is better
                    if val_score < best_score:
                        best_score = val_score
                        best_model = pipeline
                        best_scaler = scaler
                        best_model_type = model_type
                        self.scaler_type = scaler_name
            
            # Set the best model
            self.model = best_model
            self.best_model_type = best_model_type
            
            logger.info(f"Selected {best_model_type} model with {self.scaler_type} scaler")
            logger.info(f"Best validation MAE: {best_score}")
            
            # Evaluate on test set
            predictions = self.model.predict(X_test)
            
            # Calculate R² only if we have more than one sample in test set
            if len(X_test) > 1:
                r2 = r2_score(y_test, predictions)
            else:
                r2 = np.nan
                logger.warning("R² score is undefined for a single test sample. Using NaN.")
            
            metrics = {
                'mae': mean_absolute_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2,
                'actual_vs_predicted': {
                    'actual': y_test.values.tolist(),
                    'predicted': predictions.tolist()
                },
                'model_type': best_model_type,
                'scaler_type': self.scaler_type,
                'validation_mae': best_score,
                'test_years': test_years
            }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def forecast(self, df: pd.DataFrame, forecast_years: range = range(2025, 2030)) -> Dict:
        """Generate future forecasts"""
        try:
            if not self.model:
                raise ValueError("Model not trained. Call train_model() first.")
            
            if self.feature_columns is None:
                raise ValueError("Feature columns not set. Call train_model() first.")
                
            if not hasattr(self, 'feature_means'):
                raise ValueError("Feature means not available. Model might not have trained properly.")
                
            # Validate input data
            if self.target_column not in df.columns or "Year" not in df.columns:
                raise ValueError("Data must contain target column and 'Year' column")
            
            # Validate forecast years are in the future
            last_year = df["Year"].max()
            forecast_years_list = list(forecast_years)
            if min(forecast_years_list) <= last_year:
                raise ValueError("Forecast years must be in the future")
            
            # Prepare the last known data point
            last_data = df[df["Year"] == last_year].iloc[0]
            
            # Calculate average environmental conditions
            env_means = {
                'T2M_mean': df['T2M_mean'].mean(),
                'RH2M_mean': df['RH2M_mean'].mean(),
                'PRECTOTCORR_sum': df['PRECTOTCORR_sum'].mean()
            }
            
            # Initialize forecast tracking
            forecast_values = {}
            lag_features = {
                'target': [last_data[self.target_column]],
                'T2M_mean': [last_data['T2M_mean']],
                'RH2M_mean': [last_data['RH2M_mean']],
                'PRECTOTCORR_sum': [last_data['PRECTOTCORR_sum']]
            }
            
            for year in forecast_years:
                # Initialize features with stored means
                features = {col: self.feature_means.get(col, 0) for col in self.feature_columns}
                
                # Update with current environmental means
                features['T2M_mean'] = env_means['T2M_mean']
                features['RH2M_mean'] = env_means['RH2M_mean']
                features['PRECTOTCORR_sum'] = env_means['PRECTOTCORR_sum']
                
                # Update interaction features
                features['Temp_Humidity_Interaction'] = features['T2M_mean'] * features['RH2M_mean']
                features['Temp_Precip_Interaction'] = features['T2M_mean'] * features['PRECTOTCORR_sum']
                features['Humidity_Precip_Interaction'] = features['RH2M_mean'] * features['PRECTOTCORR_sum']
                
                # Update trend feature
                features['Year_Trend'] = features.get('Year_Trend', 0) + 1
                
                # Update lag features for target and environmental variables
                for lag in [1, 2]:
                    # Target lag
                    target_lag_col = f"{self.target_column}_Lag_{lag}"
                    if target_lag_col in self.feature_columns:
                        features[target_lag_col] = (
                            lag_features['target'][-lag] if len(lag_features['target']) >= lag 
                            else self.feature_means.get(target_lag_col, 0)
                        )
                    
                    # Environmental lags
                    for var in ['T2M_mean', 'RH2M_mean', 'PRECTOTCORR_sum']:
                        lag_col = f"{var}_Lag_{lag}"
                        if lag_col in self.feature_columns:
                            features[lag_col] = (
                                lag_features[var][-lag] if len(lag_features[var]) >= lag
                                else self.feature_means.get(lag_col, 0)
                            )
                
                # Update rolling statistics
                for window in [2]:
                    rolling_mean_col = f"{self.target_column}_RollingMean_{window}"
                    rolling_std_col = f"{self.target_column}_RollingStd_{window}"
                    
                    if rolling_mean_col in self.feature_columns:
                        if len(lag_features['target']) >= window:
                            features[rolling_mean_col] = np.mean(lag_features['target'][-window:])
                        else:
                            features[rolling_mean_col] = self.feature_means.get(rolling_mean_col, 0)
                    
                    if rolling_std_col in self.feature_columns:
                        if len(lag_features['target']) >= window:
                            features[rolling_std_col] = np.std(lag_features['target'][-window:])
                        else:
                            features[rolling_std_col] = self.feature_means.get(rolling_std_col, 0)
                
                # Update YoY change
                yoy_col = f"{self.target_column}_YoY"
                if yoy_col in self.feature_columns:
                    if len(lag_features['target']) >= 2:
                        features[yoy_col] = (lag_features['target'][-1] / lag_features['target'][-2] - 1)
                    else:
                        features[yoy_col] = self.feature_means.get(yoy_col, 0)
                
                # Create DataFrame with the correct column order
                features_df = pd.DataFrame([features], columns=self.feature_columns)
                
                # Make prediction
                prediction = self.model.predict(features_df)[0]
                forecast_values[str(year)] = round(prediction)
                
                # Update lag features
                lag_features['target'].append(prediction)
                for var in ['T2M_mean', 'RH2M_mean', 'PRECTOTCORR_sum']:
                    lag_features[var].append(env_means[var])
            
            return forecast_values
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            raise
            
    def save_artifacts(self, df: pd.DataFrame, metrics: Dict, forecasts: Dict):
        """Save all model artifacts"""
        try:
            if not self.model:
                raise ValueError("Model not trained. Call train_model() first.")
                
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # Save model
            if self.best_model_type == 'xgb':
                model_path = output_dir / 'malaria_model.json'
                self.model.named_steps['model'].save_model(model_path)
            else:
                import joblib
                model_path = output_dir / 'malaria_model.joblib'
                joblib.dump(self.model, model_path)
            
            # Save feature importance
            if self.best_model_type == 'xgb':
                importance = self.model.named_steps['model'].get_booster().get_score(importance_type='weight')
            elif self.best_model_type == 'rf':
                model = self.model.named_steps['model'] if hasattr(self.model, 'named_steps') else self.model
                importance = {f: i for f, i in zip(self.feature_columns, model.feature_importances_)}
            else:
                importance = None
                
            if importance:
                with open(output_dir / 'feature_importance.json', 'w') as f:
                    json.dump(importance, f, indent=4)
            
            # Save performance metrics
            metrics_path = output_dir / 'performance_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save forecasts
            forecast_path = output_dir / 'forecasts.json'
            with open(forecast_path, 'w') as f:
                json.dump(forecasts, f, indent=4)
            
            # Save visualization
            self._plot_results(df, forecasts, output_dir)
            
            logger.info(f"Artifacts saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            raise
            
    def _plot_results(self, df: pd.DataFrame, forecasts: Dict, output_dir: Path):
        """Plot actual vs predicted results"""
        try:
            if not self.model:
                raise ValueError("Model not trained. Call train_model() first.")
                
            if self.target_column not in df.columns or "Year" not in df.columns:
                raise ValueError("Data must contain target column and 'Year' column")
                
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(df['Year'], df[self.target_column], 
                    label='Historical Cases', marker='o', color='blue')
            
            # Plot forecast
            forecast_years = [int(y) for y in forecasts.keys()]
            forecast_values = list(forecasts.values())
            plt.plot(forecast_years, forecast_values, 
                    label='Forecast', marker='^', linestyle='--', color='green')
            
            # Formatting
            plt.title(f'Malaria Cases: Historical and Forecast (Model: {self.best_model_type})')
            plt.xlabel('Year')
            plt.ylabel('Total Cases')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'malaria_forecast.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise

def main():
    try:
        logger.info("Starting malaria prediction pipeline")
        
        # Initialize predictor
        predictor = MalariaPredictor(data_dir='data')
        
        # Load and prepare data
        malaria_data, env_data = predictor.load_data()
        full_data = predictor.create_features(malaria_data, env_data)
        
        # Train model
        metrics = predictor.train_model(full_data)
        
        # Generate forecasts
        forecasts = predictor.forecast(full_data)
        logger.info(f"Generated forecasts: {forecasts}")
        
        # Save all artifacts
        predictor.save_artifacts(full_data, metrics, forecasts)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()