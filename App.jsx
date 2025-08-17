import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import './App.css'

function App() {
  const [dashboardData, setDashboardData] = useState(null)
  const [dashboardMetrics, setDashboardMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  const date=new Date().getFullYear();

  useEffect(() => {
    // Load dashboard data and metrics
    Promise.all([
      fetch('/dashboard_data.json').then(res => res.json()),
      fetch('/dashboard_metrics.json').then(res => res.json())
    ])
    .then(([data, metrics]) => {
      setDashboardData(data)
      setDashboardMetrics(metrics)
      setLoading(false)
    })
    .catch(error => {
      console.error('Error loading data:', error)
      setLoading(false)
    })
  }, [])

  // Prepare chart data for Recharts
  const prepareChartData = () => {
    if (!dashboardData) return []
    
    const chartData = []
    
    // Add actual data points
    Object.entries(dashboardData.actual).forEach(([year, cases]) => {
      chartData.push({
        year: parseInt(year),
        actual: cases,
        predicted: null,
        forecasted: null,
        type: 'Historical'
      })
    })
    
    // Add predicted data point (2024)
    Object.entries(dashboardData.predicted_test).forEach(([year, cases]) => {
      const existingPoint = chartData.find(d => d.year === parseInt(year))
      if (existingPoint) {
        existingPoint.predicted = cases
        existingPoint.type = 'Validation'
      } else {
        chartData.push({
          year: parseInt(year),
          actual: null,
          predicted: cases,
          forecasted: null,
          type: 'Validation'
        })
      }
    })
    
    // Add forecasted data point (2025)
    Object.entries(dashboardData.forecasted_future).forEach(([year, cases]) => {
      chartData.push({
        year: parseInt(year),
        actual: null,
        predicted: null,
        forecasted: cases,
        type: 'Forecast'
      })
    })
    
    return chartData.sort((a, b) => a.year - b.year)
  }

  // Custom tooltip for better understanding
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="custom-tooltip">
          <h4>{`Year: ${label}`}</h4>
          {data.actual && (
            <div className="tooltip-item actual">
              <span className="tooltip-label">📊 Actual Cases:</span>
              <span className="tooltip-value">{data.actual.toLocaleString()}</span>
            </div>
          )}
          {data.predicted && (
            <div className="tooltip-item predicted">
              <span className="tooltip-label">🎯 Model Prediction:</span>
              <span className="tooltip-value">{Math.round(data.predicted).toLocaleString()}</span>
              <div className="tooltip-note">Used for model validation</div>
            </div>
          )}
          {data.forecasted && (
            <div className="tooltip-item forecasted">
              <span className="tooltip-label">🔮 Future Forecast:</span>
              <span className="tooltip-value">{Math.round(data.forecasted).toLocaleString()}</span>
              <div className="tooltip-note">Prediction for planning</div>
            </div>
          )}
          <div className="tooltip-type">
            <strong>Period:</strong> {data.type}
          </div>
        </div>
      )
    }
    return null
  }

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading Malaria Forecasting Dashboard...</p>
      </div>
    )
  }

  if (!dashboardData || !dashboardMetrics) {
    return (
      <div className="error-container">
        <h2>Error Loading Data</h2>
        <p>Unable to load dashboard data. Please try again later.</p>
      </div>
    )
  }

  const chartData = prepareChartData()

  return (
    <div className="App">
      <header className="app-header">
        <h1>🦟 Malaria Forecasting Dashboard</h1>
        <p className="subtitle">Agona West, Ghana - Interactive Predictions with Environmental Data</p>
      </header>

      <main className="dashboard-content">
        {/* Key Metrics Section */}
        <section className="metrics-section">
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>Model Accuracy</h3>
              <div className="metric-value">{dashboardMetrics.mae}</div>
              <div className="metric-label">Mean Absolute Error</div>
              <div className="metric-explanation">Lower values indicate better accuracy</div>
            </div>
            <div className="metric-card">
              <h3>Latest Year Cases</h3>
              <div className="metric-value">{dashboardMetrics.total_cases_last_year?.toLocaleString()}</div>
              <div className="metric-label">Total Cases (2024)</div>
              <div className="metric-explanation">Actual reported cases</div>
            </div>
            <div className="metric-card trend">
              <h3>Year-over-Year Change</h3>
              <div className={`metric-value ${dashboardMetrics.yoy_change < 0 ? 'positive' : 'negative'}`}>
                {dashboardMetrics.yoy_change > 0 ? '+' : ''}{dashboardMetrics.yoy_change}%
              </div>
              <div className="metric-label">vs Previous Year</div>
              <div className="metric-explanation">
                {dashboardMetrics.yoy_change < 0 ? 'Decreasing trend (Good)' : 'Increasing trend (Concerning)'}
              </div>
            </div>
          </div>
        </section>

        {/* Predictions Summary Section */}
        <section className="predictions-section">
          <h2>🎯 Model Predictions</h2>
          <div className="predictions-grid">
            <div className="prediction-card">
              <h3>2024 Model Validation</h3>
              <div className="prediction-value">{dashboardMetrics.predicted_2024?.toLocaleString()}</div>
              <div className="prediction-label">Predicted Cases</div>
              <div className="prediction-accuracy">
                <strong>Actual:</strong> {dashboardMetrics.total_cases_last_year?.toLocaleString()}
              </div>
              <div className="prediction-note">
                Model accuracy test using known data
              </div>
            </div>
            <div className="prediction-card forecast">
              <h3>2025 Future Forecast</h3>
              <div className="prediction-value">{dashboardMetrics.forecasted_2025?.toLocaleString()}</div>
              <div className="prediction-label">Forecasted Cases</div>
              <div className="prediction-accuracy">
                {((dashboardMetrics.forecasted_2025 - dashboardMetrics.total_cases_last_year) / dashboardMetrics.total_cases_last_year * 100).toFixed(1)}% vs 2024
              </div>
              <div className="prediction-note">
                Use for resource planning and intervention strategies
              </div>
            </div>
          </div>
        </section>

        {/* Interactive Chart Section */}
        <section className="chart-section">
          <h2>📈 Interactive Malaria Cases Forecast</h2>
          <div className="chart-instructions">
            <p>💡 <strong>How to use:</strong> Hover over data points for detailed information. The chart shows historical data, model validation, and future predictions.</p>
          </div>
          
          <div className="chart-container-interactive">
            <ResponsiveContainer width="100%" height={500}>
              <LineChart
                data={chartData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 60,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis 
                  dataKey="year" 
                  stroke="#666"
                  fontSize={12}
                  tickFormatter={(value) => value.toString()}
                />
                <YAxis 
                  stroke="#666"
                  fontSize={12}
                  tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="line"
                />
                
                {/* Reference line to separate historical from predictions */}
                <ReferenceLine x={2024} stroke="#ff6b6b" strokeDasharray="5 5" />
                
                {/* Actual cases line */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#1f77b4"
                  strokeWidth={4}
                  dot={{ fill: '#1f77b4', strokeWidth: 2, r: 6 }}
                  name="📊 Actual Cases (Historical)"
                  connectNulls={false}
                />
                
                {/* Predicted cases line */}
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#ff7f0e"
                  strokeWidth={4}
                  strokeDasharray="8 4"
                  dot={{ fill: '#ff7f0e', strokeWidth: 2, r: 8, stroke: '#fff' }}
                  name="🎯 Model Prediction (Validation)"
                  connectNulls={false}
                />
                
                {/* Forecasted cases line */}
                <Line
                  type="monotone"
                  dataKey="forecasted"
                  stroke="#2ca02c"
                  strokeWidth={4}
                  strokeDasharray="2 6"
                  dot={{ fill: '#2ca02c', strokeWidth: 2, r: 8, stroke: '#fff' }}
                  name="🔮 Future Forecast (Planning)"
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="chart-explanation">
            <div className="explanation-grid">
              <div className="explanation-item">
                <div className="explanation-icon">📊</div>
                <div className="explanation-content">
                  <h4>Historical Data (2021-2024)</h4>
                  <p>Actual malaria cases reported in Agona West. Shows the real trend over time.</p>
                </div>
              </div>
              <div className="explanation-item">
                <div className="explanation-icon">🎯</div>
                <div className="explanation-content">
                  <h4>Model Validation (2024)</h4>
                  <p>Tests how well our model predicts known data. Helps assess model reliability.</p>
                </div>
              </div>
              <div className="explanation-item">
                <div className="explanation-icon">🔮</div>
                <div className="explanation-content">
                  <h4>Future Forecast (2025)</h4>
                  <p>Prediction for next year based on environmental factors and historical patterns.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Clinical Insights Section */}
        <section className="insights-section">
          <h2>🏥 Clinical Insights</h2>
          <div className="insights-grid">
            <div className="insight-card">
              <h3>📉 Trend Analysis</h3>
              <p>The {dashboardMetrics.yoy_change < 0 ? 'decreasing' : 'increasing'} trend suggests that current malaria control interventions are {dashboardMetrics.yoy_change < 0 ? 'effective' : 'insufficient'}.</p>
              <div className="insight-recommendation">
                <strong>Recommendation:</strong> {dashboardMetrics.yoy_change < 0 ? 'Continue current strategies and monitor for sustained improvement.' : 'Review and strengthen intervention strategies.'}
              </div>
            </div>
            <div className="insight-card">
              <h3>📋 Resource Planning</h3>
              <p>Based on the 2025 forecast of {dashboardMetrics.forecasted_2025?.toLocaleString()} cases, plan for approximately {Math.ceil(dashboardMetrics.forecasted_2025 / 12).toLocaleString()} cases per month.</p>
              <div className="insight-recommendation">
                <strong>Action:</strong> Ensure adequate medical supplies, staff, and bed capacity for projected case load.
              </div>
            </div>
            <div className="insight-card">
              <h3>🌡️ Environmental Factors</h3>
              <p>The model incorporates temperature, humidity, and rainfall data to improve prediction accuracy for seasonal variations.</p>
              <div className="insight-recommendation">
                <strong>Note:</strong> Monitor weather patterns for early warning of potential outbreaks.
              </div>
            </div>
          </div>
        </section>

        {/* Data Summary Section */}
        <section className="data-section">
          <h2>📊 Data Summary</h2>
          <div className="data-grid">
            <div className="data-card">
              <h3>Historical Data</h3>
              <p><strong>Years:</strong> 2019-2024</p>
              <p><strong>Total Records:</strong> {Object.keys(dashboardData.actual).length} years</p>
              <p><strong>Data Source:</strong> Agona West Health Records</p>
            </div>
            <div className="data-card">
              <h3>Environmental Features</h3>
              <p><strong>Temperature:</strong> Daily averages (°C)</p>
              <p><strong>Humidity:</strong> Relative humidity (%)</p>
              <p><strong>Precipitation:</strong> Daily rainfall (mm)</p>
              <p><strong>Source:</strong> NASA POWER API</p>
            </div>
            <div className="data-card">
              <h3>Model Details</h3>
              <p><strong>Algorithm:</strong> XGBoost Regression</p>
              <p><strong>Features:</strong> Environmental + Temporal + Lag</p>
              <p><strong>Training Period:</strong> 2019-2023</p>
              <p><strong>Test Year:</strong> 2024</p>
            </div>
          </div>
        </section>

        {/* Action Buttons Section */}
        <section className="actions-section">
          <h2>🎯 Quick Actions</h2>
          <div className="actions-grid">
            <button className="action-btn primary">
              📋 Generate Report
            </button>
            <button className="action-btn secondary">
              📧 Share Forecast
            </button>
            <button className="action-btn secondary">
              ⚙️ Model Settings
            </button>
            <button className="action-btn secondary">
              📈 Export Data
            </button>
          </div>
        </section>

        {/* Technical Details Section */}
        <section className="technical-section">
          <h2>🔬 Technical Information</h2>
          <div className="technical-details">
            <div className="detail-row">
              <span className="detail-label">Model Type:</span>
              <span className="detail-value">XGBoost Regression with Environmental Features</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Data Aggregation:</span>
              <span className="detail-value">Yearly totals with environmental averages</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Feature Engineering:</span>
              <span className="detail-value">Lag features, temporal components, environmental variables</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Validation Method:</span>
              <span className="detail-value">Time series split (2024 as test year)</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Prediction Accuracy:</span>
              <span className="detail-value">MAE: {dashboardMetrics.mae} cases</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Last Updated:</span>
              <span className="detail-value">{new Date().toLocaleDateString()}</span>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>© {date} Malaria Forecasting System | Agona West Health District</p>
        <p>Powered by Machine Learning & Environmental Data</p>
      </footer>
    </div>
  )
}

export default App

