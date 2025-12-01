"""
Dashboard UI Components
Reusable visualization components for the GridSense dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime


class RiskGauge:
    """Risk meter gauge visualization."""
    
    COLORS = {
        'green': '#00FF7F',
        'yellow': '#FFD700',
        'orange': '#FFA500',
        'red': '#FF4444'
    }
    
    @staticmethod
    def create(
        value: float,
        title: str = "Grid Risk Level",
        max_value: float = 100
    ) -> go.Figure:
        """
        Create a risk gauge visualization.
        
        Args:
            value: Current risk value (0-100)
            title: Gauge title
            max_value: Maximum value for scale
        
        Returns:
            Plotly Figure object
        """
        # Determine color based on value
        if value < 50:
            color = RiskGauge.COLORS['green']
            risk_text = "LOW RISK"
        elif value < 70:
            color = RiskGauge.COLORS['yellow']
            risk_text = "MODERATE"
        elif value < 85:
            color = RiskGauge.COLORS['orange']
            risk_text = "HIGH RISK"
        else:
            color = RiskGauge.COLORS['red']
            risk_text = "CRITICAL"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{title}<br><span style='font-size:0.7em;color:{color}'>{risk_text}</span>",
                   'font': {'size': 20, 'color': '#FFFFFF'}},
            number={'font': {'size': 40, 'color': color}, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, max_value], 'tickwidth': 1, 
                        'tickcolor': "#FFFFFF", 'tickfont': {'color': '#FFFFFF'}},
                'bar': {'color': color, 'thickness': 0.75},
                'bgcolor': '#1E1E1E',
                'borderwidth': 2,
                'bordercolor': '#555',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 255, 127, 0.25)'},
                    {'range': [50, 70], 'color': 'rgba(255, 215, 0, 0.25)'},
                    {'range': [70, 85], 'color': 'rgba(255, 165, 0, 0.25)'},
                    {'range': [85, 100], 'color': 'rgba(255, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': '#FFFFFF', 'width': 3},
                    'thickness': 0.8,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='#0E1117',
            font={'color': '#FFFFFF'}
        )
        
        return fig


class LoadChart:
    """Load visualization charts."""
    
    @staticmethod
    def create_live_load(
        timestamps: List[datetime],
        loads: List[float],
        capacity: float,
        title: str = "Live Grid Load"
    ) -> go.Figure:
        """
        Create a live load chart with capacity line.
        """
        fig = go.Figure()
        
        # Load area
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=loads,
            mode='lines',
            name='Current Load',
            line=dict(color='#00D4FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.15)'
        ))
        
        # Capacity line
        fig.add_hline(
            y=capacity,
            line=dict(color='#FF4444', width=2, dash='dash'),
            annotation_text=f"Capacity: {capacity:,.0f} MW",
            annotation_position="top right",
            annotation_font_color='#FF4444'
        )
        
        # Warning threshold (90% of capacity)
        fig.add_hline(
            y=capacity * 0.9,
            line=dict(color='#FFAA00', width=1, dash='dot'),
            annotation_text="Warning (90%)",
            annotation_position="bottom right",
            annotation_font_color='#FFAA00'
        )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#FFFFFF')),
            xaxis_title="Time",
            yaxis_title="Load (MW)",
            height=400,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E2E',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#444', showgrid=True, tickfont=dict(color='#FFFFFF')),
            yaxis=dict(gridcolor='#444', showgrid=True, tickfont=dict(color='#FFFFFF')),
            legend=dict(bgcolor='rgba(30,30,46,0.8)', font=dict(color='#FFFFFF'))
        )
        
        return fig
    
    @staticmethod
    def create_load_heatmap(
        df: pd.DataFrame,
        load_col: str = 'load_mw'
    ) -> go.Figure:
        """
        Create a heatmap of load by hour and day of week.
        """
        # Prepare data
        df = df.copy()
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day'] = pd.to_datetime(df['timestamp']).dt.day_name()
        
        pivot = df.pivot_table(
            values=load_col,
            index='hour',
            columns='day',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, '#1a1a2e'],
                [0.25, '#16213e'],
                [0.5, '#0f3460'],
                [0.75, '#e94560'],
                [1, '#ff6b6b']
            ],
            colorbar=dict(
                title=dict(text='MW', font=dict(color='#FFFFFF')),
                tickfont=dict(color='#FFFFFF')
            )
        ))
        
        fig.update_layout(
            title=dict(text='Load Pattern Heatmap', font=dict(size=18, color='#FFFFFF')),
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            height=450,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E2E',
            font=dict(color='#FFFFFF'),
            xaxis=dict(tickfont=dict(color='#FFFFFF')),
            yaxis=dict(tickfont=dict(color='#FFFFFF'))
        )
        
        return fig


class ForecastPlot:
    """Forecast visualization components."""
    
    @staticmethod
    def create_forecast_chart(
        historical_timestamps: List[datetime],
        historical_loads: List[float],
        forecast_timestamps: List[datetime],
        forecast_loads: List[float],
        confidence_lower: Optional[List[float]] = None,
        confidence_upper: Optional[List[float]] = None,
        title: str = "Load Forecast"
    ) -> go.Figure:
        """
        Create a forecast chart with historical data and predictions.
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_timestamps,
            y=historical_loads,
            mode='lines',
            name='Historical',
            line=dict(color='#00D4FF', width=2)
        ))
        
        # Confidence interval
        if confidence_lower and confidence_upper:
            fig.add_trace(go.Scatter(
                x=forecast_timestamps + forecast_timestamps[::-1],
                y=list(confidence_upper) + list(confidence_lower)[::-1],
                fill='toself',
                fillcolor='rgba(255, 193, 7, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=True
            ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=forecast_loads,
            mode='lines',
            name='Forecast',
            line=dict(color='#FFC107', width=2, dash='dash')
        ))
        
        # Vertical line at forecast start
        if historical_timestamps and forecast_timestamps:
            fig.add_vline(
                x=forecast_timestamps[0],
                line=dict(color='#888', width=1, dash='dot'),
                annotation_text="Forecast Start",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#FFFFFF')),
            xaxis_title="Time",
            yaxis_title="Load (MW)",
            height=400,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E2E',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#444', tickfont=dict(color='#FFFFFF')),
            yaxis=dict(gridcolor='#444', tickfont=dict(color='#FFFFFF')),
            legend=dict(bgcolor='rgba(30,30,46,0.8)', font=dict(color='#FFFFFF'), x=0.01, y=0.99),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_scenario_comparison(
        timestamps: List[datetime],
        scenarios: Dict[str, List[float]],
        capacity: float
    ) -> go.Figure:
        """
        Compare multiple scenarios on same chart.
        """
        colors = ['#00D4FF', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
        
        fig = go.Figure()
        
        for i, (name, loads) in enumerate(scenarios.items()):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=loads,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # Capacity line
        fig.add_hline(
            y=capacity,
            line=dict(color='#FF4444', width=2, dash='dash'),
            annotation_text="Capacity",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=dict(text='Scenario Comparison', font=dict(size=18, color='#FFFFFF')),
            xaxis_title="Time",
            yaxis_title="Load (MW)",
            height=450,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E2E',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#444', tickfont=dict(color='#FFFFFF')),
            yaxis=dict(gridcolor='#444', tickfont=dict(color='#FFFFFF')),
            legend=dict(bgcolor='rgba(30,30,46,0.8)', font=dict(color='#FFFFFF'))
        )
        
        return fig


class MetricsPanel:
    """Metrics and KPI visualization."""
    
    @staticmethod
    def create_metrics_cards(metrics: Dict[str, Any]) -> str:
        """
        Generate HTML for metrics cards.
        Returns Streamlit-compatible HTML string.
        """
        cards_html = '<div style="display: flex; flex-wrap: wrap; gap: 1rem;">'
        
        for label, value in metrics.items():
            if isinstance(value, float):
                formatted = f"{value:,.2f}"
            else:
                formatted = str(value)
            
            cards_html += f'''
            <div style="
                background: linear-gradient(135deg, #1E1E2E 0%, #2D2D44 100%);
                border-radius: 10px;
                padding: 1rem;
                min-width: 150px;
                border: 1px solid #555;
            ">
                <div style="color: #CCCCCC; font-size: 0.8rem;">{label}</div>
                <div style="color: #00D4FF; font-size: 1.5rem; font-weight: bold;">{formatted}</div>
            </div>
            '''
        
        cards_html += '</div>'
        return cards_html


class RenewableChart:
    """Renewable energy visualization."""
    
    @staticmethod
    def create_renewable_mix(
        timestamps: List[datetime],
        solar: List[float],
        wind: List[float],
        total_load: List[float]
    ) -> go.Figure:
        """
        Create stacked area chart of renewable generation.
        """
        fig = go.Figure()
        
        # Solar
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=solar,
            mode='lines',
            name='Solar',
            stackgroup='renewables',
            fillcolor='rgba(255, 193, 7, 0.6)',
            line=dict(color='#FFC107', width=0)
        ))
        
        # Wind
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=wind,
            mode='lines',
            name='Wind',
            stackgroup='renewables',
            fillcolor='rgba(0, 212, 255, 0.6)',
            line=dict(color='#00D4FF', width=0)
        ))
        
        # Total load line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=total_load,
            mode='lines',
            name='Total Load',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=dict(text='Renewable Generation vs Load', font=dict(size=18, color='#FFFFFF')),
            xaxis_title="Time",
            yaxis_title="Power (MW)",
            height=400,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E2E',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#444', tickfont=dict(color='#FFFFFF')),
            yaxis=dict(gridcolor='#444', tickfont=dict(color='#FFFFFF')),
            legend=dict(bgcolor='rgba(30,30,46,0.8)', font=dict(color='#FFFFFF'))
        )
        
        return fig


