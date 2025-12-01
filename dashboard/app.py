"""
GridSense Dashboard
Main Streamlit application for load forecasting and grid simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, generate_synthetic_load_data
from data.preprocessor import DataPreprocessor
from sim.grid_simulator import GridSimulator, RiskLevel
from sim.scenarios import ScenarioManager, OutageScenario
from dashboard.components import RiskGauge, LoadChart, ForecastPlot, RenewableChart
from dashboard.report_generator import ReportGenerator


# Page configuration
st.set_page_config(
    page_title="GridSense - Smart Grid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling with light text
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 100%);
    }
    
    /* Make ALL text white/light */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #FFFFFF !important;
    }
    
    /* Headers */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #FFFFFF !important;
    }
    
    /* Main header with gradient */
    .main-header {
        background: linear-gradient(90deg, #00D4FF 0%, #7B2CBF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sub header */
    .sub-header {
        color: #CCCCCC !important;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #2D2D44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #444;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00D4FF !important;
    }
    
    .metric-label {
        color: #CCCCCC !important;
        font-size: 0.9rem;
    }
    
    /* Risk level colors */
    .risk-green { color: #00FF7F !important; }
    .risk-yellow { color: #FFD700 !important; }
    .risk-orange { color: #FFA500 !important; }
    .risk-red { color: #FF4444 !important; }
    
    /* Sidebar */
    .sidebar .sidebar-content, section[data-testid="stSidebar"] {
        background: #1E1E2E !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background-color: #2D2D44 !important;
        color: #FFFFFF !important;
        border: 1px solid #444 !important;
    }
    
    .stSelectbox label, .stSlider label, .stCheckbox label, .stTextInput label {
        color: #FFFFFF !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #00D4FF !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #AADDFF !important;
    }
    
    /* Action cards */
    .action-card {
        background: #1E1E2E;
        border-left: 4px solid #FF6B6B;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #FFFFFF !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E2E;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00D4FF !important;
    }
    
    /* Buttons */
    .stButton > button {
        color: #FFFFFF !important;
        background-color: #2D2D44 !important;
        border: 1px solid #00D4FF !important;
    }
    
    .stButton > button:hover {
        background-color: #3D3D54 !important;
        border-color: #00FFFF !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #00D4FF !important;
        color: #000000 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #FFFFFF !important;
    }
    
    /* Info, success, warning, error boxes */
    .stAlert > div {
        color: #FFFFFF !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #FFFFFF !important;
    }
    
    /* Data frames and tables */
    .stDataFrame {
        color: #FFFFFF !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        color: #FFFFFF !important;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #FFFFFF !important;
    }
    
    .stNumberInput input {
        background-color: #2D2D44 !important;
        color: #FFFFFF !important;
        border: 1px solid #444 !important;
    }
    
    /* Checkbox */
    .stCheckbox span {
        color: #FFFFFF !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #FFFFFF !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        color: #FFFFFF !important;
        background-color: #2D2D44 !important;
    }
    
    /* Write output */
    .stWrite {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'simulator' not in st.session_state:
        st.session_state.simulator = GridSimulator()
    if 'scenario_manager' not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None


def load_data():
    """Load or generate data."""
    with st.spinner("Loading data..."):
        loader = DataLoader()
        df = loader.load(source="synthetic", periods=24*30)  # 30 days
        st.session_state.df = df
        st.session_state.data_loaded = True
        return df


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        
        # Data source
        st.markdown("### 📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source",
            ["Synthetic Data", "Upload CSV", "OPSD", "ERCOT"],
            help="Choose data source for analysis"
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.data_loaded = True
        
        # Grid parameters
        st.markdown("### 🔧 Grid Parameters")
        capacity = st.slider(
            "Grid Capacity (MW)",
            min_value=5000,
            max_value=20000,
            value=10000,
            step=500
        )
        st.session_state.simulator.capacity_mw = capacity
        
        reserve_margin = st.slider(
            "Reserve Margin (%)",
            min_value=5,
            max_value=30,
            value=15
        ) / 100
        st.session_state.simulator.reserve_margin = reserve_margin
        
        # Scenario selection
        st.markdown("### 🎭 Scenario")
        scenarios = st.session_state.scenario_manager.list_scenarios()
        scenario_names = [s['name'] for s in scenarios]
        selected_scenario = st.selectbox(
            "Select Scenario",
            scenario_names,
            help="Choose simulation scenario"
        )
        
        # Custom scenario parameters
        st.markdown("### 📈 Adjustments")
        demand_growth = st.slider(
            "Demand Growth (%)",
            min_value=-10,
            max_value=30,
            value=0
        )
        
        heat_wave = st.checkbox("Heat Wave Conditions")
        industrial_spike = st.checkbox("Industrial Surge")
        renewable_fraction = st.slider(
            "Renewable Fraction (%)",
            min_value=0,
            max_value=80,
            value=20
        ) / 100
        
        # Store scenario params
        st.session_state.scenario_params = {
            'name': selected_scenario,
            'demand_growth_pct': demand_growth,
            'heat_wave': heat_wave,
            'industrial_spike': industrial_spike,
            'renewable_fraction': renewable_fraction
        }
        
        return capacity, selected_scenario


def render_main_dashboard():
    """Render main dashboard content."""
    # Header
    st.markdown('<h1 class="main-header">⚡ GridSense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Smart Grid Load Forecasting & Outage Risk Simulation</p>', unsafe_allow_html=True)
    
    # Load data if not loaded
    if not st.session_state.data_loaded:
        if st.button("🚀 Load Demo Data", use_container_width=True):
            load_data()
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        current_load = df['load_mw'].iloc[-1] if 'load_mw' in df.columns else 5000
        peak_load = df['load_mw'].max() if 'load_mw' in df.columns else 8000
        avg_load = df['load_mw'].mean() if 'load_mw' in df.columns else 5500
        capacity = st.session_state.simulator.capacity_mw
        
        with col1:
            st.metric(
                "Current Load",
                f"{current_load:,.0f} MW",
                delta=f"{(current_load/capacity*100):.1f}% of capacity"
            )
        
        with col2:
            st.metric(
                "Peak Load (24h)",
                f"{peak_load:,.0f} MW",
                delta=f"{((peak_load-avg_load)/avg_load*100):.1f}% above avg"
            )
        
        with col3:
            reserve = capacity - current_load
            st.metric(
                "Available Reserve",
                f"{reserve:,.0f} MW",
                delta=f"{(reserve/capacity*100):.1f}%"
            )
        
        with col4:
            risk_pct = (current_load / capacity) * 100
            risk_level = st.session_state.simulator.assess_risk(current_load)
            st.metric(
                "Risk Level",
                risk_level.value.upper(),
                delta=f"{risk_pct:.1f}% load factor"
            )
        
        st.markdown("---")
        
        # Main content in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Live Monitoring",
            "🔮 Forecast",
            "🎮 Simulation",
            "📄 Reports"
        ])
        
        with tab1:
            render_monitoring_tab(df)
        
        with tab2:
            render_forecast_tab(df)
        
        with tab3:
            render_simulation_tab(df)
        
        with tab4:
            render_reports_tab(df)


def render_monitoring_tab(df: pd.DataFrame):
    """Render live monitoring tab."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Live Grid Load")
        
        # Get recent data
        timestamps = pd.to_datetime(df['timestamp']).tolist()[-168:]  # Last week
        loads = df['load_mw'].tolist()[-168:]
        capacity = st.session_state.simulator.capacity_mw
        
        fig = LoadChart.create_live_load(timestamps, loads, capacity)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Risk Gauge")
        current_load = df['load_mw'].iloc[-1]
        risk_pct = (current_load / st.session_state.simulator.capacity_mw) * 100
        
        fig = RiskGauge.create(min(risk_pct, 100), "Grid Load Factor")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk indicators
        risk_level = st.session_state.simulator.assess_risk(current_load)
        
        if risk_level == RiskLevel.RED:
            st.error("🚨 CRITICAL: Immediate action required!")
        elif risk_level == RiskLevel.ORANGE:
            st.warning("⚠️ HIGH RISK: Prepare mitigation measures")
        elif risk_level == RiskLevel.YELLOW:
            st.info("📊 ELEVATED: Increased monitoring recommended")
        else:
            st.success("✅ NORMAL: Grid operating within safe limits")
    
    # Load heatmap
    st.markdown("### 🗓️ Load Pattern Analysis")
    fig = LoadChart.create_load_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_tab(df: pd.DataFrame):
    """Render forecast tab."""
    st.markdown("### 🔮 Load Forecast")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["24 Hours", "7 Days", "30 Days"]
        )
        
        model_type = st.selectbox(
            "Forecasting Model",
            ["XGBoost", "LSTM", "Prophet", "Ensemble"]
        )
        
        if st.button("🚀 Generate Forecast", use_container_width=True):
            with st.spinner(f"Training {model_type} model..."):
                # Generate simplified forecast
                hours = {"24 Hours": 24, "7 Days": 168, "30 Days": 720}[forecast_horizon]
                
                # Get last load values for trend
                last_loads = df['load_mw'].values[-168:]
                
                # Generate forecast with patterns
                np.random.seed(42)
                trend = np.mean(last_loads)
                daily_pattern = np.array([
                    trend * (0.7 + 0.3 * np.sin((h - 4) * np.pi / 12))
                    for h in range(hours)
                ])
                noise = np.random.normal(0, trend * 0.05, hours)
                forecast = daily_pattern + noise
                
                # Add growth if specified
                growth = st.session_state.scenario_params.get('demand_growth_pct', 0) / 100
                forecast *= (1 + growth)
                
                # Confidence intervals
                lower = forecast * 0.9
                upper = forecast * 1.1
                
                st.session_state.forecast_results = {
                    'values': forecast,
                    'lower': lower,
                    'upper': upper,
                    'model': model_type
                }
                
                st.success(f"✅ {model_type} forecast generated!")
    
    with col2:
        if st.session_state.forecast_results:
            forecast = st.session_state.forecast_results
            
            # Create forecast timestamps
            last_time = pd.to_datetime(df['timestamp'].iloc[-1])
            forecast_times = [last_time + timedelta(hours=i) for i in range(len(forecast['values']))]
            
            # Historical data
            hist_times = pd.to_datetime(df['timestamp']).tolist()[-168:]
            hist_loads = df['load_mw'].tolist()[-168:]
            
            fig = ForecastPlot.create_forecast_chart(
                hist_times, hist_loads,
                forecast_times, forecast['values'].tolist(),
                forecast['lower'].tolist(),
                forecast['upper'].tolist(),
                f"{forecast['model']} Load Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast metrics
            st.markdown("#### 📊 Forecast Summary")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            
            with fcol1:
                st.metric("Peak Forecast", f"{max(forecast['values']):,.0f} MW")
            with fcol2:
                st.metric("Min Forecast", f"{min(forecast['values']):,.0f} MW")
            with fcol3:
                st.metric("Average", f"{np.mean(forecast['values']):,.0f} MW")
            with fcol4:
                capacity = st.session_state.simulator.capacity_mw
                risk_hours = sum(1 for v in forecast['values'] if v > capacity * 0.9)
                st.metric("Hours at Risk", f"{risk_hours}")
        else:
            st.info("👈 Select parameters and generate forecast")


def render_simulation_tab(df: pd.DataFrame):
    """Render simulation tab."""
    st.markdown("### 🎮 Grid Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Simulation Parameters")
        
        # Get scenario
        scenario_manager = st.session_state.scenario_manager
        scenarios = scenario_manager.list_scenarios()
        scenario_names = [s['name'] for s in scenarios]
        
        selected = st.selectbox("Scenario", scenario_names, key="sim_scenario")
        
        sim_duration = st.slider("Duration (hours)", 24, 168, 72)
        
        st.markdown("#### Custom Overrides")
        custom_growth = st.number_input("Demand Growth %", -10.0, 50.0, 0.0)
        custom_heat = st.checkbox("Apply Heat Wave", key="sim_heat")
        custom_industrial = st.checkbox("Industrial Surge", key="sim_industrial")
        
        run_sim = st.button("▶️ Run Simulation", use_container_width=True, type="primary")
    
    with col2:
        if run_sim:
            with st.spinner("Running simulation..."):
                # Get base load forecast
                base_loads = df['load_mw'].values[-sim_duration:] if len(df) >= sim_duration else df['load_mw'].values
                
                # Pad if needed
                if len(base_loads) < sim_duration:
                    base_loads = np.tile(base_loads, int(np.ceil(sim_duration / len(base_loads))))[:sim_duration]
                
                # Run simulation
                simulator = st.session_state.simulator
                simulator.reset()
                
                start_time = datetime.now()
                
                result = simulator.simulate_scenario(
                    base_loads,
                    start_time,
                    scenario_params={
                        'name': selected,
                        'demand_growth_pct': custom_growth,
                        'heat_wave': custom_heat,
                        'industrial_spike': custom_industrial,
                        'renewable_fraction': 0.2
                    }
                )
                
                st.session_state.simulation_result = result
                
                # Display results
                st.success("✅ Simulation complete!")
                
                # Risk timeline
                timestamps = result.timestamps
                loads = result.loads
                
                fig = LoadChart.create_live_load(
                    timestamps, loads,
                    simulator.capacity_mw,
                    "Simulated Load Profile"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results summary
                st.markdown("#### 📊 Simulation Results")
                
                rcol1, rcol2, rcol3 = st.columns(3)
                
                with rcol1:
                    st.metric("Peak Load", f"{result.peak_load:,.0f} MW")
                    st.metric("Min Reserve", f"{result.min_reserve_margin*100:.1f}%")
                
                with rcol2:
                    st.metric("Outage Events", len(result.outage_events))
                    st.metric("Shedding Events", len(result.load_shedding_events))
                
                with rcol3:
                    st.metric("Unserved Energy", f"{result.total_unserved_energy_mwh:,.0f} MWh")
                    red_hours = sum(1 for r in result.risk_levels if r == RiskLevel.RED)
                    st.metric("Critical Hours", red_hours)
                
                # Mitigation actions
                if result.mitigation_actions:
                    st.markdown("#### 🛡️ Recommended Actions")
                    for action in result.mitigation_actions:
                        st.markdown(f'<div class="action-card">{action}</div>', unsafe_allow_html=True)


def render_reports_tab(df: pd.DataFrame):
    """Render reports tab."""
    st.markdown("### 📄 Report Generation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Report Options")
        
        report_title = st.text_input("Report Title", "GridSense Analysis Report")
        
        include_forecast = st.checkbox("Include Forecast Analysis", value=True)
        include_simulation = st.checkbox("Include Simulation Results", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Generating report..."):
                generator = ReportGenerator()
                
                # Prepare summary
                summary = {
                    "Analysis Period": f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "N/A",
                    "Total Data Points": len(df),
                    "Average Load (MW)": df['load_mw'].mean() if 'load_mw' in df.columns else 0,
                    "Peak Load (MW)": df['load_mw'].max() if 'load_mw' in df.columns else 0,
                    "Min Load (MW)": df['load_mw'].min() if 'load_mw' in df.columns else 0,
                    "Grid Capacity (MW)": st.session_state.simulator.capacity_mw
                }
                
                # Prepare sections
                forecast_data = None
                if include_forecast and st.session_state.forecast_results:
                    forecast = st.session_state.forecast_results
                    forecast_data = {
                        "Model Used": forecast['model'],
                        "Forecast Peak (MW)": max(forecast['values']),
                        "Forecast Average (MW)": np.mean(forecast['values']),
                        "Hours at Risk": sum(1 for v in forecast['values'] 
                                           if v > st.session_state.simulator.capacity_mw * 0.9)
                    }
                
                sim_results = None
                if include_simulation and hasattr(st.session_state, 'simulation_result'):
                    result = st.session_state.simulation_result
                    sim_results = {
                        "Scenario": result.scenario_name,
                        "Duration (hours)": result.duration_hours,
                        "Peak Load (MW)": result.peak_load,
                        "Minimum Reserve Margin": f"{result.min_reserve_margin*100:.1f}%",
                        "Outage Events": len(result.outage_events),
                        "Unserved Energy (MWh)": result.total_unserved_energy_mwh
                    }
                
                recommendations = None
                if include_recommendations:
                    recommendations = [
                        "Monitor grid load during peak hours (9 AM - 9 PM)",
                        "Ensure reserve margin stays above 15% threshold",
                        "Pre-position demand response resources for high-risk periods",
                        "Coordinate with neighboring utilities for emergency imports",
                        "Maintain real-time communication with large industrial customers"
                    ]
                    if hasattr(st.session_state, 'simulation_result'):
                        recommendations.extend(st.session_state.simulation_result.mitigation_actions[:3])
                
                # Generate report
                pdf_content = generator.generate_report(
                    report_title,
                    summary,
                    forecast_data,
                    sim_results,
                    recommendations
                )
                
                st.session_state.pdf_content = pdf_content
                st.success("✅ Report generated!")
    
    with col2:
        if hasattr(st.session_state, 'pdf_content'):
            st.markdown("#### 📄 Report Preview")
            
            # Show summary
            st.info("Report generated successfully! Click below to download.")
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=st.session_state.pdf_content,
                file_name=f"gridsense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            # Quick stats
            st.markdown("#### Quick Summary")
            if st.session_state.df is not None:
                df = st.session_state.df
                st.write(f"- **Data Points:** {len(df):,}")
                st.write(f"- **Peak Load:** {df['load_mw'].max():,.0f} MW")
                st.write(f"- **Average Load:** {df['load_mw'].mean():,.0f} MW")
                st.write(f"- **Grid Capacity:** {st.session_state.simulator.capacity_mw:,} MW")
        else:
            st.info("👈 Configure and generate report")


def render_footer():
    """Render footer with attribution."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #888;">
        <p style="font-size: 1rem; margin-bottom: 0.5rem;">
            ⚡ <strong style="color: #00D4FF;">GridSense</strong> - Smart Grid Load Forecasting System
        </p>
        <p style="font-size: 0.9rem; color: #AAAAAA;">
            Made by <strong style="color: #FFFFFF;">Jalal Diab</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_main_dashboard()
    render_footer()


if __name__ == "__main__":
    main()


