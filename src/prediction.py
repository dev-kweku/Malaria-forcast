import pandas as pd
import numpy as np
from models.model_training import MalariaModel

class MalariaPredictor:
    def __init__(self, model_path):
        self.model = MalariaModel.load_model(model_path)
        self.age_groups = list(self.model.age_mapping.keys())
        self.sex_options = ['Male', 'Female']
    
    def predict_for_period(self, start_year, end_year, sex, age_group):
        """Predict for a range of years"""
        years = list(range(start_year, end_year + 1))
        predictions = self.model.predict_future(years, sex, age_group)
        
        # Convert to DataFrame
        result_df = pd.DataFrame({
            'Year': list(predictions.keys()),
            'Predicted Positivity Rate': list(predictions.values())
        })
        
        return result_df
    
    def predict_all_demographics(self, year):
        """Predict for all demographic groups for a specific year"""
        results = []
        
        for sex in self.sex_options:
            for age_group in self.age_groups:
                if age_group not in ['Total Male', 'Total Female', 'Total']:
                    prediction = self.model.predict_future([year], sex, age_group)
                    results.append({
                        'Year': year,
                        'Sex': sex,
                        'Age Group': age_group,
                        'Predicted Positivity Rate': prediction[year]
                    })
        
        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    predictor = MalariaPredictor('../models/malaria_model.pkl')
    
    # Predict for specific demographic
    result = predictor.predict_for_period(2025, 2029, 'Male', '20-34')
    print(result)
    
    # Predict for all demographics in 2025
    all_results = predictor.predict_all_demographics(2025)
    print(all_results.head())