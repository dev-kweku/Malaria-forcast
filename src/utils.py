import pandas as pd
import numpy as np

def validate_inputs(year, sex, age_group):
    """Validate user inputs"""
    if not (2025 <= year <= 2035):
        raise ValueError("Year must be between 2025 and 2035")
    
    if sex not in ['Male', 'Female']:
        raise ValueError("Sex must be either 'Male' or 'Female'")
    
    valid_age_groups = [
        '<28days', '1-11mths', '1-4', '5-9', '10-14',
        '15-17', '18-19', '20-34', '35-49', '50-59',
        '60-69', '70+'
    ]
    
    if age_group not in valid_age_groups:
        raise ValueError("Invalid age group")
    
    return True

def format_predictions(predictions_dict):
    """Format predictions for display"""
    formatted = {}
    for year, rate in predictions_dict.items():
        formatted[year] = f"{rate:.2%}"
    return formatted

def calculate_confidence_interval(predictions, confidence=0.95):
    """Calculate confidence intervals for predictions"""
    mean = np.mean(list(predictions.values()))
    std = np.std(list(predictions.values()))
    
    # Z-score for 95% confidence
    z_score = 1.96
    
    margin_of_error = z_score * (std / np.sqrt(len(predictions)))
    
    return {
        'mean': mean,
        'lower_bound': max(0, mean - margin_of_error),
        'upper_bound': min(1, mean + margin_of_error)
    }