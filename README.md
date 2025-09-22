# Malaria Prediction Dashboard - Complete Documentation

## Overview

The Malaria Prediction Dashboard is a Streamlit-based web application designed to predict malaria positivity rates for future years based on historical data from Agona West, Ghana. The application provides interactive visualizations, demographic analysis, and trend predictions to help public health officials and researchers understand and anticipate malaria patterns.

## Table of Contents

1. [Application Structure](#application-structure)
2. [Installation & Setup](#installation--setup)
3. [Model Architecture](#model-architecture)
4. [User Interface Components](#user-interface-components)
5. [Functionality](#functionality)
6. [Data Requirements](#data-requirements)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting](#troubleshooting)
9. [Deployment Options](#deployment-options)

## Application Structure

```
malaria-dashboard/
├── app.py                 # Main Streamlit application
├── models/
│   └── malaria_model.pkl  # Trained prediction model
├── data/
│   └── CLEANEDDATA.xlsx   # Historical malaria data
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone or download the application files**
   ```bash
   git clone <repository-url>
   cd malaria-dashboard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is not available, install dependencies manually:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn openpyxl
   ```

4. **Prepare the data and model**
   - Place historical data in `data/CLEANEDDATA.xlsx`
   - Ensure the trained model is at `models/malaria_model.pkl`

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Model Architecture

### Model Components
The prediction model (`malaria_model.pkl`) contains:
- **Trained Model**: A scikit-learn regression model (likely RandomForest or Gradient Boosting)
- **Age Mapping**: Dictionary encoding age groups to numerical values
- **Sex Mapping**: Dictionary encoding gender to numerical values
- **Feature Names**: List of features used during training

### Prediction Methodology
The model uses three input features:
1. **Year**: The prediction year
2. **Sex**: Encoded gender (Male/Female)
3. **Age Group**: Encoded age category

The output is a predicted positivity rate between 0 and 1.

### Model Training
To retrain the model (if needed):
1. Ensure historical data is available in the correct format
2. Run the training script (if available)
3. The script will generate a new `malaria_model.pkl` file

## User Interface Components

### Main Dashboard
- **Header**: Application title and brief description
- **Prediction Controls**: Sidebar with input parameters
- **Visualization Area**: Main content with charts and data displays

### Sidebar Components
1. **Prediction Period**: Start and end year selectors (2025-2035)
2. **Demographic Selection**: Sex and age group dropdowns
3. **Additional Analysis**: Year selector for demographic analysis
4. **Historical Data**: Toggle to show historical trends
5. **Export Functionality**: Button to download predictions

### Visualization Elements
- **Line Charts**: Show trend predictions over time
- **Treemaps**: Display demographic distribution of predictions
- **Heatmaps**: Show positivity rates across age and gender
- **Data Tables**: Present numerical results in tabular format
- **Metrics Cards**: Highlight key statistics (average, min, max rates)

## Functionality

### Core Features

1. **Yearly Predictions**
   - Predict positivity rates for a selected range of years
   - Visualize trends with interactive line charts
   - Display numerical results in a sortable table

2. **Demographic Analysis**
   - Analyze predictions across all age and gender groups
   - Visualize distributions using treemaps and heatmaps
   - Compare susceptibility across demographic segments

3. **Historical Data Visualization**
   - View historical trends from 2019-2024
   - Compare male vs. female positivity rates over time
   - Examine yearly summary statistics

4. **Data Export**
   - Download prediction results as CSV files
   - Share data for further analysis outside the application

### Technical Implementation

- **Caching**: Uses `@st.cache_resource` for efficient model loading
- **Error Handling**: Comprehensive try-catch blocks throughout
- **Responsive Design**: Adapts to different screen sizes
- **Interactive Elements**: Plotly charts with hover information

## Data Requirements

### Historical Data Format
The application expects historical data in `data/CLEANEDDATA.xlsx` with the following columns:

| Column Name | Description | Data Type |
|-------------|-------------|-----------|
| YEAR | Year of data record | Integer |
| SEX | Gender (Male/Female) | String |
| SUSPECTED | Number of suspected cases | Integer |
| SUSPECTED TESTED | Number of tested cases | Integer |
| POSITIVE | Number of positive cases | Integer |

### Data Processing
The application automatically:
- Identifies the header row in Excel files
- Filters for only Male/Female gender values
- Calculates positivity rate (POSITIVE / SUSPECTED TESTED)
- Handles missing or invalid data points

## Usage Guide

### Making Predictions

1. **Set Prediction Parameters**
   - Select start and end years (2025-2035)
   - Choose a demographic group (sex and age)
   - Click "Generate Prediction"

2. **Interpret Results**
   - Examine the trend line for increasing/decreasing patterns
   - Review the numerical predictions in the data table
   - Note the average, maximum, and minimum rates

3. **Compare Demographics**
   - Select a year for demographic analysis
   - Click "Analyze Demographics"
   - Compare rates across age groups and genders

### Analyzing Historical Trends

1. **Enable Historical Data**
   - Check "Show Historical Data Trends" in the sidebar
   - View historical patterns from 2019-2024

2. **Compare with Predictions**
   - Note historical trends when making future predictions
   - Identify seasonal patterns or long-term trends

### Exporting Data

1. **Generate Export File**
   - Click "Export Predictions" in the sidebar
   - Download the generated CSV file

2. **Use Exported Data**
   - Import into statistical software for further analysis
   - Create custom visualizations
   - Combine with other health data sources

## Troubleshooting

### Common Issues

1. **Model Not Found Error**
   - Ensure `malaria_model.pkl` exists in the models folder
   - Retrain the model if necessary

2. **Historical Data Not Loading**
   - Verify `CLEANEDDATA.xlsx` is in the data folder
   - Check that the Excel file has the required columns

3. **Dependency Issues**
   - Confirm all required packages are installed
   - Check Python version compatibility

4. **Performance Problems**
   - The application may be slow with very large historical datasets
   - Consider sampling historical data if performance is inadequate

### Error Messages

- **"Model not trained yet!"**: The model file is missing or invalid
- **"File not found"**: Historical data file is missing
- **"Missing columns in historical data"**: Excel file doesn't have required columns
- **"Prediction error"**: Issue with generating predictions from the model

## Deployment Options

### Local Deployment
1. Run with Streamlit: `streamlit run src/dashboared.py`
2. Access at: `http://localhost:8501`

### Cloud Deployment
1. **Streamlit Sharing**
   - Push code to GitHub repository
   - Connect repository to Streamlit Sharing
   - Deploy automatically


### Performance Considerations
- The application is optimized for datasets of moderate size
- For very large historical datasets, consider implementing data sampling
- Model loading is cached for better performance

## Maintenance

### Regular Updates
1. **Update Dependencies**: Periodically update Python packages
2. **Retrain Model**: Refresh model with new data as it becomes available
3. **Verify Data**: Ensure historical data format remains consistent

### Model Retraining
To maintain prediction accuracy:
1. Collect new historical data regularly
2. Retrain the model with updated datasets
3. Validate model performance before deployment

