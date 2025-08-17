import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import xgboost as xgb
import joblib
from datetime import datetime
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Malaria Prediction Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #212529;
    }
    .chart-container {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .chart-container:hover {
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #6c757d;
        font-size: 0.9rem;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .tab-content {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 5px;
        transition: background-color 0.3s;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    .model-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        color: #333;
    }
    .data-card {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        color: #333;
    }
    .forecast-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model():
    """Load the trained model and artifacts"""
    output_dir = Path('output')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    if (output_dir / 'malaria_model.json').exists():
        model = xgb.XGBRegressor()
        model.load_model(output_dir / 'malaria_model.json')
        model_type = 'XGBoost'
    elif (output_dir / 'malaria_model.joblib').exists():
        model = joblib.load(output_dir / 'malaria_model.joblib')
        model_type = 'Random Forest'
    else:
        st.error("Model file not found. Please run the training script first.")
        return None, None, None, None, None, None
    
    # Load metrics
    if (output_dir / 'performance_metrics.json').exists():
        with open(output_dir / 'performance_metrics.json', 'r') as f:
            metrics = json.load(f)
    else:
        st.error("Metrics file not found. Please run the training script first.")
        return None, None, None, None, None, None
    
    # Load forecasts
    if (output_dir / 'forecasts.json').exists():
        with open(output_dir / 'forecasts.json', 'r') as f:
            forecasts = json.load(f)
    else:
        st.error("Forecasts file not found. Please run the training script first.")
        return None, None, None, None, None, None
    
    # Load feature importance
    if (output_dir / 'feature_importance.json').exists():
        with open(output_dir / 'feature_importance.json', 'r') as f:
            feature_importance = json.load(f)
    else:
        feature_importance = None
    
    # Load historical data
    if (output_dir / 'historical_data.csv').exists():
        historical_data = pd.read_csv(output_dir / 'historical_data.csv')
    else:
        historical_data = None
    
    return model, metrics, forecasts, feature_importance, historical_data, model_type

# Load all data
model, metrics, forecasts, feature_importance, historical_data, model_type = load_model()

# Check if data loaded successfully
if model is None:
    st.stop()

# Main header
st.markdown('<h1 class="main-header">🦟 Malaria Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar with model information
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Model Type:** {model_type}")
st.sidebar.markdown(f"**Scaler:** {metrics.get('scaler_type', 'Standard')}")
st.sidebar.markdown(f"**Validation MAE:** {metrics.get('validation_mae', 'N/A'):.2f}")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Check for R² NaN and show warning
if pd.isna(metrics.get('r2', np.nan)):
    st.markdown("""
    <div class="warning-box">
        <strong>Warning:</strong> R² score is not available (NaN). This typically happens when the test set contains only one sample. 
        Consider using more data for testing or adjusting the train/test split.
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Model Analysis", "📈 Forecast", "📁 Data Explorer"])

with tab1:
    # Key metrics
    st.markdown("### Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Mean Absolute Error</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["mae"]:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Root Mean Squared Error</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["rmse"]:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">R² Score</div>', unsafe_allow_html=True)
        r2_value = metrics.get('r2', np.nan)
        if pd.isna(r2_value):
            r2_display = "N/A"
        else:
            r2_display = f"{r2_value:.2f}"
        st.markdown(f'<div class="metric-value">{r2_display}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Model Type</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{model_type}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Forecast section
    st.markdown("### Malaria Cases Forecast")
    st.markdown('<div class="chart-container forecast-card">', unsafe_allow_html=True)

    # Prepare forecast data
    forecast_years = [int(year) for year in forecasts.keys()]
    forecast_values = [int(value) for value in forecasts.values()]

    # Create forecast plot
    fig = go.Figure()

    # Add historical data if available
    if historical_data is not None:
        fig.add_trace(go.Scatter(
            x=historical_data['Year'],
            y=historical_data['Total Cases'],
            mode='lines+markers',
            name='Historical Cases',
            line=dict(color='royalblue', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>Year</b>: %{x}<br><b>Cases</b>: %{y:,}<extra></extra>'
        ))

    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='firebrick', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>Year</b>: %{x}<br><b>Forecast</b>: %{y:,}<extra></extra>'
    ))

    # Add vertical line to separate historical and forecast
    if historical_data is not None:
        last_historical_year = historical_data['Year'].max()
        fig.add_vline(x=last_historical_year + 0.5, line_width=2, line_dash="dash", line_color="gray")
        fig.add_annotation(
            x=last_historical_year + 0.5,
            y=max(historical_data['Total Cases'].max(), max(forecast_values)) * 0.9,
            text="Forecast Start",
            showarrow=False,
            textangle=-90,
            xanchor="right",
            font=dict(size=12, color="gray")
        )

    # Update layout
    fig.update_layout(
        title="Malaria Cases: Historical Data and Forecast",
        xaxis_title="Year",
        yaxis_title="Total Cases",
        hovermode='x unified',
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Forecast table with download button
    st.markdown("### Forecast Details")
    st.markdown('<div class="chart-container forecast-card">', unsafe_allow_html=True)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Year': forecast_years,
        'Predicted Cases': forecast_values,
        'Year-over-Year Change': [''] + [
            f"{((forecast_values[i] - forecast_values[i-1]) / forecast_values[i-1] * 100):.1f}%"
            for i in range(1, len(forecast_values))
        ]
    })

    # Format numbers
    forecast_df['Predicted Cases'] = forecast_df['Predicted Cases'].map('{:,}'.format)

    # Display dataframe
    st.dataframe(forecast_df, use_container_width=True)

    # Download button
    csv = forecast_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="malaria_forecast.csv"><button class="download-button">Download Forecast Data</button></a>'
    st.markdown(href, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Feature importance section
    if feature_importance:
        st.markdown("### Feature Importance Analysis")
        st.markdown('<div class="chart-container model-card">', unsafe_allow_html=True)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            importance_df.head(15),  # Show top 15 features
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Important Features',
            color='Importance',
            color_continuous_scale='Viridis',
            hover_data=['Importance'],
            labels={'Importance': 'Feature Importance Score'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            template="plotly_white",
            xaxis_title="Feature Importance Score",
            yaxis_title="Feature"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Actual vs Predicted section
    st.markdown("### Model Performance Analysis")
    st.markdown('<div class="chart-container model-card">', unsafe_allow_html=True)

    actual_values = metrics['actual_vs_predicted']['actual']
    predicted_values = metrics['actual_vs_predicted']['predicted']

    # Create figure
    fig_actual_pred = go.Figure()
    fig_actual_pred.add_trace(go.Scatter(
        x=actual_values,
        y=predicted_values,
        mode='markers',
        name='Actual vs Predicted',
        marker=dict(size=12, color='royalblue', opacity=0.7),
        hovertemplate='<b>Actual</b>: %{x:,}<br><b>Predicted</b>: %{y:,}<extra></extra>'
    ))

    # Add perfect prediction line
    min_val = min(min(actual_values), min(predicted_values))
    max_val = max(max(actual_values), max(predicted_values))
    fig_actual_pred.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='firebrick', dash='dash')
    ))

    fig_actual_pred.update_layout(
        title="Actual vs Predicted Values Comparison Test Set",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        hovermode='closest',
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_actual_pred, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Residuals plot
    st.markdown("### Residual Analysis")
    st.markdown('<div class="chart-container model-card">', unsafe_allow_html=True)

    # Calculate residuals
    residuals = np.array(actual_values) - np.array(predicted_values)

    # Create residuals plot
    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(
        x=predicted_values,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(size=10, color='green', opacity=0.7),
        hovertemplate='<b>Predicted</b>: %{x:,}<br><b>Residual</b>: %{y:,}<extra></extra>'
    ))

    # Add zero line
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

    fig_residuals.update_layout(
        title="Residuals vs Predicted Values",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        hovermode='closest',
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_residuals, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # Interactive forecast controls
    st.markdown("### Interactive Forecast Controls")
    st.markdown('<div class="chart-container forecast-card">', unsafe_allow_html=True)

    # Year range slider
    min_year = min(forecast_years)
    max_year = max(forecast_years)
    selected_years = st.slider(
        "Select Forecast Years",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Filter forecast data
    filtered_forecasts = {str(year): forecasts[str(year)] for year in range(selected_years[0], selected_years[1] + 1)}

    # Create interactive forecast plot
    fig = go.Figure()

    # Add historical data if available
    if historical_data is not None:
        fig.add_trace(go.Scatter(
            x=historical_data['Year'],
            y=historical_data['Total Cases'],
            mode='lines+markers',
            name='Historical Cases',
            line=dict(color='royalblue', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>Year</b>: %{x}<br><b>Cases</b>: %{y:,}<extra></extra>'
        ))

    # Add forecast data
    filtered_years = [int(year) for year in filtered_forecasts.keys()]
    filtered_values = [int(value) for value in filtered_forecasts.values()]
    fig.add_trace(go.Scatter(
        x=filtered_years,
        y=filtered_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='firebrick', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>Year</b>: %{x}<br><b>Forecast</b>: %{y:,}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f"Malaria Cases Forecast ({selected_years[0]}-{selected_years[1]})",
        xaxis_title="Year",
        yaxis_title="Total Cases",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Forecast confidence intervals (simulated)
    st.markdown("### Forecast with Confidence Intervals")
    st.markdown('<div class="chart-container forecast-card">', unsafe_allow_html=True)

    # Simulate confidence intervals (±10% of forecast)
    upper_bound = [int(value * 1.1) for value in filtered_values]
    lower_bound = [int(value * 0.9) for value in filtered_values]

    # Create plot with confidence intervals
    fig = go.Figure()

    # Add historical data if available
    if historical_data is not None:
        fig.add_trace(go.Scatter(
            x=historical_data['Year'],
            y=historical_data['Total Cases'],
            mode='lines+markers',
            name='Historical Cases',
            line=dict(color='royalblue', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>Year</b>: %{x}<br><b>Cases</b>: %{y:,}<extra></extra>'
        ))

    # Add forecast data
    fig.add_trace(go.Scatter(
        x=filtered_years,
        y=filtered_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='firebrick', width=3),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>Year</b>: %{x}<br><b>Forecast</b>: %{y:,}<extra></extra>'
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=filtered_years + filtered_years[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='90% Confidence Interval'
    ))

    fig.update_layout(
        title="Malaria Cases Forecast with Confidence Intervals",
        xaxis_title="Year",
        yaxis_title="Total Cases",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    # Data explorer section
    st.markdown("### Historical Data Explorer")
    st.markdown('<div class="chart-container data-card">', unsafe_allow_html=True)

    if historical_data is not None:
        # Display historical data
        st.dataframe(historical_data, use_container_width=True)
        
        # Download button for historical data
        csv = historical_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="historical_data.csv"><button class="download-button">Download Historical Data</button></a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Historical data not available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Environmental factors over time
    st.markdown("### Environmental Factors Over Time")
    st.markdown('<div class="chart-container data-card">', unsafe_allow_html=True)

    if historical_data is not None:
        # Create subplot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature', 'Humidity', 'Precipitation'),
            vertical_spacing=0.08
        )

        # Temperature
        fig.add_trace(
            go.Scatter(
                x=historical_data['Year'],
                y=historical_data['T2M_mean'],
                name='Temperature',
                line=dict(color='red'),
                hovertemplate='<b>Year</b>: %{x}<br><b>Temp</b>: %{y:.1f}°C<extra></extra>'
            ),
            row=1, col=1
        )

        # Humidity
        fig.add_trace(
            go.Scatter(
                x=historical_data['Year'],
                y=historical_data['RH2M_mean'],
                name='Humidity',
                line=dict(color='blue'),
                hovertemplate='<b>Year</b>: %{x}<br><b>Humidity</b>: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Precipitation
        fig.add_trace(
            go.Scatter(
                x=historical_data['Year'],
                y=historical_data['PRECTOTCORR_sum'],
                name='Precipitation',
                line=dict(color='green'),
                hovertemplate='<b>Year</b>: %{x}<br><b>Precip</b>: %{y:.1f}mm<extra></extra>'
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=800,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("### Feature Correlation Heatmap")
    st.markdown('<div class="chart-container data-card">', unsafe_allow_html=True)

    if historical_data is not None:
        # Select numeric columns for correlation
        numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
        corr_matrix = historical_data[numeric_cols].corr()

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix"
        )

        fig.update_layout(
            height=600,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with project information
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
### Malaria Prediction System - Final Year Project

**Project Description**: This project develops a machine learning system to predict malaria cases based on environmental factors. 
The system uses historical malaria data and environmental variables to forecast future malaria outbreaks, 
enabling proactive public health interventions.

**Technologies Used**: Python, Streamlit, XGBoost, Scikit-learn, Plotly, Pandas

**Features**:
- Advanced machine learning models for prediction
- Interactive data visualization
- Real-time forecasting with confidence intervals
- Comprehensive model analysis and evaluation
- Data exploration tools

**Author**: Your Name
**Supervisor**: Dr. Supervisor Name
**Institution**: Your University
**Date**: {}
""".format(datetime.now().strftime("%Y-%m-%d")))
st.markdown('</div>', unsafe_allow_html=True)