# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pickle
# import warnings
# import sys
# import os
# warnings.filterwarnings('ignore')

# # Add the src directory to the path so we can import malaria_model
# sys.path.append(os.path.dirname(__file__))

# try:
#     from malaria_model import MalariaModel
# except ImportError:
#     st.error("Could not import MalariaModel. Make sure malaria_model.py is in the src directory.")

# # Set page config
# st.set_page_config(
#     page_title="Malaria Prediction Dashboard",
#     page_icon="🦟",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .prediction-card {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#         margin-bottom: 20px;
#         border-left: 4px solid #1f77b4;
#     }
#     .metric-card {
#         background-color: #ffffff;
#         padding: 15px;
#         border-radius: 8px;
#         border: 1px solid #e0e0e0;
#         text-align: center;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
# </style>
# """, unsafe_allow_html=True)

# class MalariaPredictor:
#     def __init__(self, model_path):
#         try:
#             with open(model_path, 'rb') as f:
#                 self.model = pickle.load(f)
#             self.age_groups = list(self.model.age_mapping.keys())
#             self.sex_options = ['Male', 'Female']
#             st.success("✅ Model loaded successfully!")
#         except FileNotFoundError:
#             st.warning("⚠️ Model file not found. Please train the model first.")
#             self.model = None
#         except Exception as e:
#             st.error(f"❌ Error loading model: {e}")
#             self.model = None
    
#     def predict_for_period(self, start_year, end_year, sex, age_group):
#         """Predict for a range of years"""
#         if self.model is None:
#             return pd.DataFrame()
        
#         years = list(range(start_year, end_year + 1))
#         predictions = self.model.predict_future(years, sex, age_group)
        
#         # Convert to DataFrame
#         result_df = pd.DataFrame({
#             'Year': list(predictions.keys()),
#             'Predicted Positivity Rate': list(predictions.values())
#         })
        
#         return result_df
    
#     def predict_all_demographics(self, year):
#         """Predict for all demographic groups for a specific year"""
#         if self.model is None:
#             return pd.DataFrame()
        
#         results = []
        
#         for sex in self.sex_options:
#             for age_group in self.age_groups:
#                 prediction = self.model.predict_future([year], sex, age_group)
#                 results.append({
#                     'Year': year,
#                     'Sex': sex,
#                     'Age Group': age_group,
#                     'Predicted Positivity Rate': prediction[year]
#                 })
        
#         return pd.DataFrame(results)

# def load_historical_data(file_path):
#     """Load and process historical data"""
#     try:
#         # Read Excel file and find header row
#         df_raw = pd.read_excel(file_path, header=None)
        
#         # Find the header row (look for row with 'YEAR')
#         header_row = None
#         for i in range(min(10, len(df_raw))):
#             row_values = [str(x).strip() for x in df_raw.iloc[i].values if pd.notna(x)]
#             if 'YEAR' in row_values:
#                 header_row = i
#                 break
        
#         if header_row is None:
#             st.error("Could not find header row in historical data")
#             return pd.DataFrame()
        
#         # Read with correct header
#         df = pd.read_excel(file_path, header=header_row)
#         df.columns = [str(col).strip() for col in df.columns]
        
#         # Filter and clean data
#         df = df[df['SEX'].isin(['Male', 'Female'])]
        
#         # Convert numeric columns
#         numeric_cols = ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']
#         for col in numeric_cols:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         df = df.dropna(subset=numeric_cols)
#         df['POSITIVITY_RATE'] = df['POSITIVE'] / df['SUSPECTED TESTED']
#         df = df[(df['POSITIVITY_RATE'] >= 0) & (df['POSITIVITY_RATE'] <= 1)]
        
#         return df
        
#     except Exception as e:
#         st.error(f"Error loading historical data: {e}")
#         return pd.DataFrame()

# # Header
# st.markdown('<h1 class="main-header">🦟 Malaria Prediction Dashboard</h1>', unsafe_allow_html=True)
# st.markdown("""
# This dashboard predicts malaria positivity rates for future years based on historical data from Agona West.
# Use the controls in the sidebar to configure predictions.
# """)

# # Initialize predictor
# @st.cache_resource
# def load_predictor():
#     model_path = 'models/malaria_model.pkl'
#     if os.path.exists(model_path):
#         return MalariaPredictor(model_path)
#     else:
#         st.warning("Model not found. Please train the model first.")
#         return MalariaPredictor(None)

# predictor = load_predictor()


# # Sidebar
# st.sidebar.header("📊 Prediction Parameters")

# # Year selection
# st.sidebar.subheader("Prediction Period")
# start_year = st.sidebar.number_input("Start Year", min_value=2025, max_value=2035, value=2025)
# end_year = st.sidebar.number_input("End Year", min_value=2025, max_value=2035, value=2029)

# # Demographic selection
# st.sidebar.subheader("Demographic Selection")
# sex = st.sidebar.selectbox("Sex", predictor.sex_options if predictor.model else ['Male', 'Female'])
# age_group = st.sidebar.selectbox("Age Group", predictor.age_groups if predictor.model else [
#     '<28days', '1-11mths', '1-4', '5-9', '10-14', '15-17', '18-19', 
#     '20-34', '35-49', '50-59', '60-69', '70+'
# ])

# # Main content
# if predictor.model is None:
#     st.warning("""
#     ⚠️ **Model not trained yet!**
    
#     Please run the model training script first:
#     ```bash
#     python models/model_training.py
#     ```
    
#     This will create the malaria prediction model that this dashboard uses.
#     """)
# else:
#     # Generate prediction
#     if st.sidebar.button("🚀 Generate Prediction", type="primary"):
#         with st.spinner("Generating predictions..."):
#             # Get prediction
#             prediction_df = predictor.predict_for_period(start_year, end_year, sex, age_group)
            
#             if not prediction_df.empty:
#                 # Display results
#                 st.subheader(f"📈 Prediction for {sex}, {age_group} ({start_year}-{end_year})")
                
#                 # Create two columns for layout
#                 col1, col2 = st.columns([1, 2])
                
#                 with col1:
#                     st.markdown("### 📋 Prediction Results")
#                     styled_df = prediction_df.copy()
#                     styled_df['Predicted Positivity Rate'] = styled_df['Predicted Positivity Rate'].apply(lambda x: f"{x:.2%}")
#                     st.dataframe(styled_df, use_container_width=True)
                    
#                     # Metrics
#                     avg_rate = prediction_df['Predicted Positivity Rate'].mean()
#                     max_rate = prediction_df['Predicted Positivity Rate'].max()
#                     min_rate = prediction_df['Predicted Positivity Rate'].min()
                    
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Average Rate", f"{avg_rate:.2%}")
#                     with col2:
#                         st.metric("Highest Rate", f"{max_rate:.2%}")
#                     with col3:
#                         st.metric("Lowest Rate", f"{min_rate:.2%}")
                
#                 with col2:
#                     # Create visualization
#                     fig = px.line(
#                         prediction_df, 
#                         x='Year', 
#                         y='Predicted Positivity Rate',
#                         title=f'Malaria Positivity Rate Prediction for {sex}, {age_group}',
#                         markers=True,
#                         line_shape='spline'
#                     )
#                     fig.update_traces(line=dict(width=3), marker=dict(size=8))
#                     fig.update_layout(
#                         yaxis_tickformat='.0%',
#                         hovermode='x unified',
#                         plot_bgcolor='rgba(0,0,0,0)',
#                         yaxis_title="Positivity Rate",
#                         xaxis_title="Year"
#                     )
#                     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
#                     fig.update_xaxes(showgrid=False)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 # Show prediction details
#                 st.markdown("### 📊 Prediction Insights")
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("""
#                     **Trend Analysis:**
#                     - 📈 Increasing trend: Rates are rising over time
#                     - 📉 Decreasing trend: Rates are falling over time
#                     - ➡️ Stable trend: Rates remain relatively constant
#                     """)
                
#                 with col2:
#                     trend = "stable"
#                     if len(prediction_df) > 1:
#                         first_rate = prediction_df['Predicted Positivity Rate'].iloc[0]
#                         last_rate = prediction_df['Predicted Positivity Rate'].iloc[-1]
#                         if last_rate > first_rate + 0.05:
#                             trend = "increasing 📈"
#                         elif last_rate < first_rate - 0.05:
#                             trend = "decreasing 📉"
#                         else:
#                             trend = "stable ➡️"
                    
#                     st.metric("Overall Trend", trend)

# # Additional analysis section
# st.sidebar.header("📈 Additional Analysis")
# analysis_year = st.sidebar.number_input("Year for Demographic Analysis", min_value=2025, max_value=2035, value=2025)

# if st.sidebar.button("🔍 Analyze Demographics") and predictor.model:
#     with st.spinner("Analyzing demographic trends..."):
#         # Get predictions for all demographics
#         all_demographics = predictor.predict_all_demographics(analysis_year)
        
#         if not all_demographics.empty:
#             # Display results
#             st.subheader(f"👥 Demographic Analysis for {analysis_year}")
            
#             # Create visualization
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 fig = px.treemap(
#                     all_demographics, 
#                     path=['Sex', 'Age Group'], 
#                     values='Predicted Positivity Rate',
#                     title=f'Malaria Positivity Rate by Demographic ({analysis_year})',
#                     color='Predicted Positivity Rate',
#                     color_continuous_scale='Reds'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
            
#             with col2:
#                 # Heatmap
#                 heatmap_data = all_demographics.pivot_table(
#                     values='Predicted Positivity Rate', 
#                     index='Age Group', 
#                     columns='Sex'
#                 )
                
#                 fig = px.imshow(
#                     heatmap_data,
#                     title=f'Positivity Rate Heatmap ({analysis_year})',
#                     aspect='auto',
#                     color_continuous_scale='Reds'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
            
#             # Show data table
#             st.markdown("### 📊 Detailed Demographic Predictions")
#             styled_demo = all_demographics.copy()
#             styled_demo['Predicted Positivity Rate'] = styled_demo['Predicted Positivity Rate'].apply(lambda x: f"{x:.2%}")
#             st.dataframe(styled_demo, use_container_width=True)

# # Historical data section
# st.sidebar.header("📚 Historical Data")
# show_historical = st.sidebar.checkbox("Show Historical Data Trends")

# if show_historical:
#     st.subheader("📊 Historical Data Trends (2019-2024)")
    
#     # Load historical data
#     historical_df = load_historical_data('data/CLEANEDDATA.xlsx')
    
#     if not historical_df.empty:
#         # Group by year and calculate average positivity rate
#         yearly_trend = historical_df.groupby('YEAR')['POSITIVITY_RATE'].mean().reset_index()
        
#         # Create visualization
#         col1, col2 = st.columns(2)
        
#         with col1:
#             fig = px.line(
#                 yearly_trend, 
#                 x='YEAR', 
#                 y='POSITIVITY_RATE',
#                 title='Historical Malaria Positivity Rate Trend',
#                 markers=True,
#                 line_shape='spline'
#             )
#             fig.update_layout(
#                 yaxis_tickformat='.0%',
#                 xaxis_title="Year",
#                 yaxis_title="Average Positivity Rate"
#             )
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             # Yearly comparison by sex
#             sex_trend = historical_df.groupby(['YEAR', 'SEX'])['POSITIVITY_RATE'].mean().reset_index()
#             fig = px.line(
#                 sex_trend,
#                 x='YEAR',
#                 y='POSITIVITY_RATE',
#                 color='SEX',
#                 title='Positivity Rate by Sex (Historical)',
#                 markers=True
#             )
#             fig.update_layout(yaxis_tickformat='.0%')
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Show data table
#         st.markdown("### 📋 Historical Summary Statistics")
#         yearly_stats = historical_df.groupby('YEAR').agg({
#             'SUSPECTED': 'sum',
#             'SUSPECTED TESTED': 'sum',
#             'POSITIVE': 'sum',
#             'POSITIVITY_RATE': 'mean'
#         }).reset_index()
        
#         yearly_stats['POSITIVITY_RATE'] = yearly_stats['POSITIVITY_RATE'].apply(lambda x: f"{x:.2%}")
#         st.dataframe(yearly_stats, use_container_width=True)
#     else:
#         st.warning("Could not load historical data. Please check the data file.")

# # Footer
# st.markdown("---")
# st.markdown("""
# **Malaria Prediction Dashboard** | Final Year Project | Agona West Malaria Trends  
# *Built with Streamlit, Scikit-learn, and Plotly*
# """)

# # Add download functionality
# if predictor.model and st.sidebar.button("💾 Export Predictions"):
#     try:
#         # Generate sample data for export
#         predictions_2025 = predictor.predict_all_demographics(2025)
#         predictions_2025['Predicted Positivity Rate'] = predictions_2025['Predicted Positivity Rate'].apply(lambda x: f"{x:.2%}")
        
#         # Convert to CSV
#         csv = predictions_2025.to_csv(index=False)
#         st.sidebar.download_button(
#             label="Download Predictions (CSV)",
#             data=csv,
#             file_name="malaria_predictions_2025.csv",
#             mime="text/csv"
#         )
#     except Exception as e:
#         st.sidebar.error(f"Error exporting data: {e}")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Malaria Prediction Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dataframe th, .dataframe td {
        text-align: center !important;
    }
    .dataframe td:nth-child(1) {
        font-weight: 600;
    }
    .period-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 500;
    }
    .chart-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Model wrapper + predictor
# -------------------------
class MalariaModelWrapper:
    def __init__(self, model_data):
        # model_data expected to be a dict with keys: 'model', 'age_mapping', 'sex_mapping', 'feature_names'
        self.model = model_data.get('model', None)
        self.age_mapping = model_data.get('age_mapping', {}) or {}
        self.sex_mapping = model_data.get('sex_mapping', {}) or {}
        self.feature_names = model_data.get('feature_names', []) or []

        # Determine expected number of features safely
        if hasattr(self.model, 'named_steps'):
            first_step = list(self.model.named_steps.values())[0]
            self.n_features = getattr(first_step, 'n_features_in_', 3)
        else:
            self.n_features = getattr(self.model, 'n_features_in_', 3)

    def predict_future(self, years, sex, age_group):
        """Make predictions for future years. Returns dict year -> rate (0..1)."""
        if self.model is None:
            return {}

        try:
            sex_encoded = self.sex_mapping.get(sex, 0)
            age_encoded = self.age_mapping.get(age_group, 0)
            predictions = {}

            for year in years:
                if self.n_features == 3:
                    features = np.array([[year, sex_encoded, age_encoded]])
                elif self.n_features == 6:
                    years_since_start = year - 2019
                    year_age_interaction = year * age_encoded
                    year_sex_interaction = year * sex_encoded
                    features = np.array([[year, sex_encoded, age_encoded, years_since_start, year_age_interaction, year_sex_interaction]])
                else:
                    features = np.array([[year, sex_encoded, age_encoded]])

                # model.predict should accept a 2D array
                prediction = self.model.predict(features)[0]
                predictions[int(year)] = float(max(0.0, min(prediction, 1.0)))
            return predictions

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return {}

class MalariaPredictor:
    def __init__(self, model_path=None):
        # If model_path is None or doesn't exist, set defaults and model=None
        default_age_groups = ['<28days', '1-11mths', '1-4', '5-9', '10-14', '15-17', '18-19', '20-34', '35-49', '50-59', '60-69', '70+']
        default_sex = ['Male', 'Female']

        if not model_path or not os.path.exists(model_path):
            # Model missing — use defaults
            self.model = None
            self.age_groups = default_age_groups
            self.sex_options = default_sex
            # Only show a warning elsewhere (so page load isn't spammed repeatedly)
            return

        # Try loading the model file
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = MalariaModelWrapper(model_data)

            # Safely get available mappings (fallback to defaults)
            age_map_keys = list(getattr(self.model, 'age_mapping', {}).keys())
            sex_map_keys = list(getattr(self.model, 'sex_mapping', {}).keys())

            self.age_groups = age_map_keys if age_map_keys else default_age_groups
            self.sex_options = sex_map_keys if sex_map_keys else default_sex

            st.success(f"✅ Model loaded successfully! (Expects {self.model.n_features} features)")
        except FileNotFoundError:
            st.warning("⚠️ Model file not found. Please train and place the model at the specified path.")
            self.model = None
            self.age_groups = default_age_groups
            self.sex_options = default_sex
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None
            self.age_groups = default_age_groups
            self.sex_options = default_sex

    def predict_for_period(self, start_year, end_year, sex, age_group):
        if self.model is None:
            return pd.DataFrame()

        try:
            start_year = int(start_year)
            end_year = int(end_year)
            if start_year > end_year:
                st.error("Start year cannot be greater than end year")
                return pd.DataFrame()

            years = list(range(start_year, end_year + 1))
            predictions = self.model.predict_future(years, sex, age_group)

            # Convert to sorted DataFrame
            result_df = pd.DataFrame({
                'Year': [int(y) for y in sorted(predictions.keys())],
                'Predicted Positivity Rate': [predictions[y] for y in sorted(predictions.keys())]
            })
            return result_df

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return pd.DataFrame()

    def predict_all_demographics(self, year):
        if self.model is None:
            return pd.DataFrame()

        try:
            results = []
            for sex in self.sex_options:
                for age_group in self.age_groups:
                    prediction = self.model.predict_future([int(year)], sex, age_group)
                    if int(year) in prediction:
                        results.append({
                            'Year': int(year),
                            'Sex': sex,
                            'Age Group': age_group,
                            'Predicted Positivity Rate': prediction[int(year)]
                        })
            return pd.DataFrame(results)
        except Exception as e:
            st.error(f"Demographic prediction error: {e}")
            return pd.DataFrame()

# -------------------------
# Historical data loader
# -------------------------
def load_historical_data(file_path):
    """Load and process historical excel data; tries to detect header row."""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()

        # Read without header to search for header row
        df_raw = pd.read_excel(file_path, header=None)
        header_row = None
        for i in range(min(10, len(df_raw))):
            row_values = [str(x).strip().upper() for x in df_raw.iloc[i].values if pd.notna(x)]
            if 'YEAR' in row_values:
                header_row = i
                break

        if header_row is None:
            st.error("Could not find header row in historical data")
            return pd.DataFrame()

        df = pd.read_excel(file_path, header=header_row)
        df.columns = [str(col).strip().upper() for col in df.columns]

        # Normalize expected names
        # Accept either 'SUSPECTED' or 'SUSPECTED TESTED'
        if 'SUSPECTED TESTED' not in df.columns and 'SUSPECTED' in df.columns:
            df['SUSPECTED TESTED'] = df['SUSPECTED']

        required_cols = ['YEAR', 'SEX', 'SUSPECTED TESTED', 'POSITIVE']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in historical data: {missing_cols}")
            return pd.DataFrame()

        # Filter for Male/Female rows only
        df = df[df['SEX'].isin(['Male', 'Female'])]

        # Convert numeric columns
        for col in ['YEAR', 'SUSPECTED', 'SUSPECTED TESTED', 'POSITIVE']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['YEAR', 'SUSPECTED TESTED', 'POSITIVE'])
        df['YEAR'] = df['YEAR'].astype(int)
        df['POSITIVITY_RATE'] = df['POSITIVE'] / df['SUSPECTED TESTED']
        df = df[(df['POSITIVITY_RATE'] >= 0) & (df['POSITIVITY_RATE'] <= 1)]

        return df

    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()

# -------------------------
# App UI
# -------------------------
st.markdown('<h1 class="main-header">🦟 Malaria Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard predicts malaria positivity rates for future years based on historical data from Agona West.
Use the controls in the sidebar to configure predictions.
""")

# Load predictor (caches between runs)
@st.cache_resource
def load_predictor():
    model_path = 'models/malaria_model.pkl'
    # Return MalariaPredictor — it will handle missing path gracefully
    return MalariaPredictor(model_path if os.path.exists(model_path) else None)

predictor = load_predictor()

# Sidebar controls
st.sidebar.header("📊 Prediction Parameters")
st.sidebar.subheader("Prediction Period")
st.sidebar.markdown('<div class="period-info">Fixed Period: 2025-2029</div>', unsafe_allow_html=True)
start_year, end_year = 2025, 2029

st.sidebar.subheader("Demographic Selection")
sex_options = getattr(predictor, 'sex_options', ['Male', 'Female'])
age_groups = getattr(predictor, 'age_groups', ['<28days', '1-11mths', '1-4', '5-9', '10-14', '15-17', '18-19',
                                                '20-34', '35-49', '50-59', '60-69', '70+'])
sex = st.sidebar.selectbox("Sex", sex_options)
age_group = st.sidebar.selectbox("Age Group", age_groups)

# Show model missing message if necessary
if predictor.model is None:
    st.warning("""
    ⚠️ **Model not trained yet!**
    
    Please run the model training script first to create `models/malaria_model.pkl` or place the model file in that path.
    """)

# Generate prediction button (only does something if a model exists)
if st.sidebar.button("🚀 Generate Prediction", type="primary"):
    if predictor.model is None:
        st.error("No model available. Cannot generate predictions.")
    else:
        with st.spinner("Generating predictions..."):
            prediction_df = predictor.predict_for_period(start_year, end_year, sex, age_group)

            if prediction_df.empty:
                st.error("Failed to generate predictions. Please check the model and inputs.")
            else:
                # Header and layout
                st.subheader(f"📈 Prediction for {sex}, {age_group} ({start_year}-{end_year})")
                left_col, right_col = st.columns([1, 2])

                # Left column: table + metrics
                with left_col:
                    st.markdown("### 📋 Prediction Results (table)")
                    # Display: convert Year to string only for display
                    display_df = prediction_df.copy()
                    display_df_display = display_df.copy()
                    display_df_display['Year'] = display_df_display['Year'].astype(str)
                    display_df_display['Predicted Positivity Rate'] = display_df_display['Predicted Positivity Rate'].apply(lambda x: f"{x:.2%}")

                    # Show dataframe (simple)
                    st.dataframe(display_df_display, use_container_width=True, hide_index=True)

                    # Metrics
                    avg_rate = prediction_df['Predicted Positivity Rate'].mean()
                    max_rate = prediction_df['Predicted Positivity Rate'].max()
                    min_rate = prediction_df['Predicted Positivity Rate'].min()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Average Rate", f"{avg_rate:.2%}")
                    m2.metric("Highest Rate", f"{max_rate:.2%}")
                    m3.metric("Lowest Rate", f"{min_rate:.2%}")

                # Right column: improved plot
                with right_col:
                    st.markdown("### 📊 Prediction Chart")
                    # Ensure Year is integer
                    prediction_df['Year'] = prediction_df['Year'].astype(int)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=prediction_df['Year'],
                        y=prediction_df['Predicted Positivity Rate'],
                        mode='lines+markers',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=10, color='#1f77b4', line=dict(width=1.5, color='white')),
                        hovertemplate='<b>Year</b>: %{x}<br><b>Positivity Rate</b>: %{y:.2%}<extra></extra>',
                        name='Predicted Rate'
                    ))

                    # Y-axis range: scale up a bit from the max; keep a small floor if extremely low values
                    y_max = prediction_df['Predicted Positivity Rate'].max()
                    y_upper = max(y_max * 1.2, 0.05) if y_max > 0 else 0.1
                    fig.update_yaxes(title='Positivity Rate', tickformat='.0%', gridcolor='rgba(0,0,0,0.1)', range=[0, y_upper])

                    fig.update_xaxes(title='Year', tickvals=prediction_df['Year'], ticktext=prediction_df['Year'].astype(str), gridcolor='rgba(0,0,0,0.1)')

                    # Add clear annotations above each point
                    for x, y in zip(prediction_df['Year'], prediction_df['Predicted Positivity Rate']):
                        fig.add_annotation(
                            x=int(x),
                            y=float(y),
                            text=f"{y:.2%}",
                            showarrow=False,
                            yshift=12,
                            font=dict(size=11, color='#1f77b4'),
                            bgcolor='white',
                            bordercolor='#1f77b4',
                            borderwidth=1,
                            borderpad=3,
                            opacity=0.95
                        )

                    fig.update_layout(
                        title=dict(
                            text=f'Malaria Positivity Rate Prediction<br><span style=\"font-size:14px;color:#666\">{sex}, {age_group} ({start_year}-{end_year})</span>',
                            x=0.5
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(l=40, r=40, t=90, b=60),
                        height=450,
                        hovermode='x unified',
                        showlegend=False
                    )

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Prediction insights
                st.markdown("### 📊 Prediction Insights")
                insight_col1, insight_col2 = st.columns(2)
                with insight_col1:
                    st.markdown("""
                    **Trend Analysis:**
                    - 📈 Increasing trend: Rates are rising over time
                    - 📉 Decreasing trend: Rates are falling over time
                    - ➡️ Stable trend: Rates remain relatively constant
                    """)

                with insight_col2:
                    # Simple linear trend on the predicted series
                    if len(prediction_df) > 1:
                        x = np.arange(len(prediction_df))
                        y = prediction_df['Predicted Positivity Rate'].values
                        slope, intercept = np.polyfit(x, y, 1)
                        if abs(slope) < 0.005:
                            trend = "stable ➡️"
                        elif slope > 0:
                            trend = "increasing 📈"
                        else:
                            trend = "decreasing 📉"
                    else:
                        trend = "insufficient data"

                    st.metric("Overall Trend", trend)

# -------------------------
# Additional demographic analysis
# -------------------------
st.sidebar.header("📈 Additional Analysis")
analysis_year = st.sidebar.number_input("Year for Demographic Analysis", min_value=2025, max_value=2029, value=2025, format="%d")

if st.sidebar.button("🔍 Analyze Demographics"):
    if predictor.model is None:
        st.error("No model available for demographic analysis.")
    else:
        with st.spinner("Analyzing demographic trends..."):
            all_demographics = predictor.predict_all_demographics(int(analysis_year))

            if all_demographics.empty:
                st.error("Failed to generate demographic predictions.")
            else:
                st.subheader(f"👥 Demographic Analysis for {analysis_year}")
                c1, c2 = st.columns(2)

                # Treemap
                with c1:
                    try:
                        fig = px.treemap(
                            all_demographics,
                            path=['Sex', 'Age Group'],
                            values='Predicted Positivity Rate',
                            title=f'Malaria Positivity Rate by Demographic ({analysis_year})',
                            color='Predicted Positivity Rate',
                            color_continuous_scale='Reds'
                        )
                        # Force percentages in colorbar ticks is handled by plotly formatting; acceptable as-is.
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create treemap: {e}")

                # Heatmap
                with c2:
                    try:
                        heatmap_data = all_demographics.pivot_table(
                            values='Predicted Positivity Rate',
                            index='Age Group',
                            columns='Sex'
                        )

                        if heatmap_data.empty or heatmap_data.shape[0] < 1 or heatmap_data.shape[1] < 1:
                            st.warning("Insufficient data for heatmap visualization")
                        else:
                            fig = px.imshow(
                                heatmap_data,
                                title=f'Positivity Rate Heatmap ({analysis_year})',
                                aspect='auto',
                                color_continuous_scale='Reds',
                                labels=dict(color="Positivity Rate"),
                                text_auto=True
                            )
                            fig.update_traces(texttemplate="%{z:.2%}")
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create heatmap: {e}")

                # Table of demographic predictions (formatted)
                st.markdown("### 📊 Detailed Demographic Predictions")
                styled_demo = all_demographics.copy()
                styled_demo['Predicted Positivity Rate'] = styled_demo['Predicted Positivity Rate'].apply(lambda x: f"{x:.2%}")
                styled_demo['Year'] = styled_demo['Year'].astype(str)
                st.dataframe(styled_demo, use_container_width=True, hide_index=True)

# -------------------------
# Historical data display
# -------------------------
st.sidebar.header("📚 Historical Data")
show_historical = st.sidebar.checkbox("Show Historical Data Trends")

if show_historical:
    st.subheader("📊 Historical Data Trends (2019-2024)")
    historical_df = load_historical_data('data/CLEANEDDATA.xlsx')

    if historical_df.empty:
        st.warning("Could not load historical data. Please check the data file path and format.")
    else:
        # Yearly average positivity
        yearly_trend = historical_df.groupby('YEAR')['POSITIVITY_RATE'].mean().reset_index().sort_values('YEAR')

        colh1, colh2 = st.columns(2)
        with colh1:
            fig = px.line(
                yearly_trend,
                x='YEAR',
                y='POSITIVITY_RATE',
                title='Historical Malaria Positivity Rate Trend',
                markers=True,
                line_shape='spline'
            )
            fig.update_layout(yaxis_tickformat='.0%', xaxis_title="Year", yaxis_title="Average Positivity Rate", xaxis=dict(type='category'))
            st.plotly_chart(fig, use_container_width=True)

        with colh2:
            sex_trend = historical_df.groupby(['YEAR', 'SEX'])['POSITIVITY_RATE'].mean().reset_index()
            fig2 = px.line(
                sex_trend,
                x='YEAR',
                y='POSITIVITY_RATE',
                color='SEX',
                title='Positivity Rate by Sex (Historical)',
                markers=True
            )
            fig2.update_layout(yaxis_tickformat='.0%', xaxis=dict(type='category'))
            st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.markdown("### 📋 Historical Summary Statistics")
        yearly_stats = historical_df.groupby('YEAR').agg({
            'SUSPECTED': 'sum' if 'SUSPECTED' in historical_df.columns else ('SUSPECTED TESTED' if 'SUSPECTED TESTED' in historical_df.columns else 'sum'),
            'SUSPECTED TESTED': 'sum',
            'POSITIVE': 'sum',
            'POSITIVITY_RATE': 'mean'
        }).reset_index()

        # Some columns may be missing depending on file - coerce safe formatting
        # Ensure we only format existing columns
        if 'POSITIVITY_RATE' in yearly_stats.columns:
            yearly_stats['POSITIVITY_RATE'] = yearly_stats['POSITIVITY_RATE'].apply(lambda x: f"{x:.2%}")
        yearly_stats['YEAR'] = yearly_stats['YEAR'].astype(str)
        st.dataframe(yearly_stats, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
**Malaria Prediction Dashboard** | Final Year Project | Agona West Malaria Trends  
*Built with Streamlit, Scikit-learn, and Plotly*
""")

# Export functionality (download CSV)
if predictor.model is not None and st.sidebar.button("💾 Export Predictions"):
    try:
        predictions_2025 = predictor.predict_all_demographics(2025)
        if predictions_2025.empty:
            st.sidebar.error("No predictions available to export.")
        else:
            export_df = predictions_2025.copy()
            export_df['Predicted Positivity Rate'] = export_df['Predicted Positivity Rate'].apply(lambda x: f"{x:.2%}")
            export_df['Year'] = export_df['Year'].astype(int)
            csv = export_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download Predictions (CSV)",
                data=csv,
                file_name="malaria_predictions_2025.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.sidebar.error(f"Error exporting data: {e}")
