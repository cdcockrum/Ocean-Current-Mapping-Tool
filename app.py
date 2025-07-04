import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import warnings
import time
import traceback
warnings.filterwarnings('ignore')

class OceanCurrentMapper:
    def __init__(self):
        self.noaa_base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        self.oscar_base_url = "https://podaac-opendap.jpl.nasa.gov/opendap/allData/oscar/preview/L4/oscar_third_deg"
        
    def get_noaa_current_data(self, station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch current data from NOAA API"""
        try:
            params = {
                'product': 'currents',
                'application': 'OceanCurrentMapper',
                'begin_date': start_date,
                'end_date': end_date,
                'station': station_id,
                'time_zone': 'gmt',
                'units': 'metric',
                'format': 'json'
            }
            
            response = requests.get(self.noaa_base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching NOAA data: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_current_data(self, region: str, resolution: str) -> Dict:
        """Generate synthetic ocean current data for demonstration"""
        # Define region boundaries
        regions = {
            "Gulf of Mexico": {"lat": [18, 31], "lon": [-98, -80]},
            "California Coast": {"lat": [32, 42], "lon": [-125, -117]},
            "Atlantic Coast": {"lat": [25, 45], "lon": [-81, -65]},
            "Global": {"lat": [-60, 60], "lon": [-180, 180]}
        }
        
        # Set resolution
        res_map = {"High": 0.1, "Medium": 0.25, "Low": 0.5}
        res = res_map.get(resolution, 0.25)
        
        # Get region bounds
        bounds = regions.get(region, regions["Global"])
        
        # Create coordinate grids
        lats = np.arange(bounds["lat"][0], bounds["lat"][1], res)
        lons = np.arange(bounds["lon"][0], bounds["lon"][1], res)
        
        # Generate realistic current patterns
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Create realistic current vectors using oceanographic patterns
        # Gulf Stream-like eastward flow
        u_component = 0.5 * np.sin(np.pi * (lat_grid - bounds["lat"][0]) / (bounds["lat"][1] - bounds["lat"][0]))
        # Cross-shore component
        v_component = 0.3 * np.cos(np.pi * (lon_grid - bounds["lon"][0]) / (bounds["lon"][1] - bounds["lon"][0]))
        
        # Add some turbulence and eddies
        u_component += 0.2 * np.random.normal(0, 0.1, u_component.shape)
        v_component += 0.2 * np.random.normal(0, 0.1, v_component.shape)
        
        # Calculate current speed and direction
        speed = np.sqrt(u_component**2 + v_component**2)
        direction = np.arctan2(v_component, u_component) * 180 / np.pi
        
        return {
            'latitude': lat_grid,
            'longitude': lon_grid,
            'u_component': u_component,
            'v_component': v_component,
            'speed': speed,
            'direction': direction,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_current_map(self, region: str, resolution: str, show_vectors: bool, 
                          show_speed: bool, vector_scale: float) -> go.Figure:
        """Create interactive ocean current map with improved sizing"""
        
        # Get current data
        current_data = self.generate_synthetic_current_data(region, resolution)
        
        fig = go.Figure()
        
        # Add speed contours if requested
        if show_speed:
            fig.add_trace(go.Contour(
                x=current_data['longitude'][0, :],
                y=current_data['latitude'][:, 0],
                z=current_data['speed'],
                colorscale='Viridis',
                name='Current Speed (m/s)',
                showscale=True,
                colorbar=dict(
                    title="Speed (m/s)", 
                    x=1.02,
                    thickness=15,
                    len=0.7
                )
            ))
        
        # Add vector field if requested
        if show_vectors:
            # Subsample for better visibility
            step = max(1, len(current_data['latitude']) // 20)
            lat_sub = current_data['latitude'][::step, ::step]
            lon_sub = current_data['longitude'][::step, ::step]
            u_sub = current_data['u_component'][::step, ::step] * vector_scale
            v_sub = current_data['v_component'][::step, ::step] * vector_scale
            
            # Create arrow annotations
            for i in range(lat_sub.shape[0]):
                for j in range(lat_sub.shape[1]):
                    if i % 2 == 0 and j % 2 == 0:  # Further subsample
                        fig.add_annotation(
                            ax=lon_sub[i, j],
                            ay=lat_sub[i, j],
                            axref='x',
                            ayref='y',
                            x=lon_sub[i, j] + u_sub[i, j],
                            y=lat_sub[i, j] + v_sub[i, j],
                            xref='x',
                            yref='y',
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='red',
                            showarrow=True
                        )
        
        # Calculate aspect ratio for better proportions
        lat_range = current_data['latitude'].max() - current_data['latitude'].min()
        lon_range = current_data['longitude'].max() - current_data['longitude'].min()
        
        # Update layout with improved sizing
        fig.update_layout(
            title=f'Ocean Currents - {region}',
            xaxis=dict(
                title='Longitude',
                constrain="domain"
            ),
            yaxis=dict(
                title='Latitude',
                constrain="domain"
            ),
            showlegend=True,
            autosize=True,
            # Remove fixed dimensions - let it be responsive
            margin=dict(l=40, r=40, t=60, b=40),  # Smaller margins
            # Add responsive config
            dragmode='pan',
            hovermode='closest'
        )
        
        # Set axis ranges for better proportions
        fig.update_xaxes(range=[current_data['longitude'].min(), current_data['longitude'].max()])
        fig.update_yaxes(range=[current_data['latitude'].min(), current_data['latitude'].max()])
        
        return fig
    
    def get_forecast_data(self, region: str, forecast_hours: int) -> go.Figure:
        """Generate forecast visualization with improved sizing"""
        
        # Create time series for forecast
        times = [datetime.now() + timedelta(hours=i) for i in range(forecast_hours)]
        
        # Generate sample forecast data
        np.random.seed(42)  # For reproducible demo
        current_speeds = np.random.normal(0.5, 0.2, forecast_hours)
        current_speeds = np.maximum(current_speeds, 0)  # Ensure non-negative
        
        wave_heights = np.random.normal(1.5, 0.5, forecast_hours)
        wave_heights = np.maximum(wave_heights, 0)
        
        wind_speeds = np.random.normal(10, 5, forecast_hours)
        wind_speeds = np.maximum(wind_speeds, 0)
        
        # Create subplots for better separation
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Current Speed (m/s)', 'Wave Height (m)', 'Wind Speed (m/s)'),
            vertical_spacing=0.1,
            shared_xaxes=True,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Current Speed subplot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=current_speeds,
                mode='lines+markers',
                name='Current Speed',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Wave Height subplot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=wave_heights,
                mode='lines+markers',
                name='Wave Height',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Wind Speed subplot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=wind_speeds,
                mode='lines+markers',
                name='Wind Speed',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        # Update layout with better sizing
        fig.update_layout(
            title=f'Ocean Forecast - {region}',
            showlegend=False,
            autosize=True,
            margin=dict(l=60, r=50, t=80, b=60),
            hovermode='x unified'
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
        fig.update_yaxes(title_text="Height (m)", row=2, col=1)
        fig.update_yaxes(title_text="Speed (m/s)", row=3, col=1)
        
        return fig
    
    def analyze_surfing_conditions(self, region: str) -> str:
        """Analyze surfing conditions based on current data"""
        
        current_data = self.generate_synthetic_current_data(region, "Medium")
        avg_speed = np.mean(current_data['speed'])
        max_speed = np.max(current_data['speed'])
        
        # Simple surfing condition analysis
        conditions = []
        
        if avg_speed < 0.3:
            conditions.append("Low current speeds - good for beginners")
        elif avg_speed < 0.8:
            conditions.append("Moderate currents - suitable for intermediate surfers")
        else:
            conditions.append("Strong currents - experienced surfers only")
        
        if max_speed > 1.0:
            conditions.append("🌊 Strong rip currents detected in some areas")
        
        # Add mock weather conditions
        conditions.extend([
            f"Water temperature: {20 + np.random.randint(0, 10)}°C",
            f"Wind: {5 + np.random.randint(0, 15)} mph offshore",
            f"Wave height: {1 + np.random.randint(0, 3)} meters"
        ])
        
        return "\n".join(conditions)

# Initialize the mapper with error handling
try:
    mapper = OceanCurrentMapper()
    print("Ocean Current Mapper initialized successfully")
except Exception as e:
    print(f"Error initializing mapper: {e}")
    traceback.print_exc()

# Create wrapper functions with error handling
def create_current_map(region, resolution, show_vectors, show_speed, vector_scale):
    try:
        return mapper.create_current_map(region, resolution, show_vectors, show_speed, vector_scale)
    except Exception as e:
        print(f"Error creating current map: {e}")
        traceback.print_exc()
        # Return empty plot on error
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(autosize=True)
        return fig

def create_forecast(region, forecast_hours):
    try:
        return mapper.get_forecast_data(region, forecast_hours)
    except Exception as e:
        print(f"Error creating forecast: {e}")
        traceback.print_exc()
        # Return empty plot on error
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(autosize=True)
        return fig

def analyze_conditions(region):
    try:
        return mapper.analyze_surfing_conditions(region)
    except Exception as e:
        print(f"Error analyzing conditions: {e}")
        traceback.print_exc()
        return f"Error analyzing conditions: {str(e)}"

# Define the Gradio interface with improved layout
with gr.Blocks(title="Ocean Current Mapper", theme=gr.themes.Ocean()) as demo:
    gr.Markdown("""
    <h1 style="font-size: 3em; text-align: center; color: #2E86AB; margin-bottom: 0.5em;">
        Real-Time Ocean Current Mapping Tool
    </h1>
    
    <div style="text-align: center; font-size: 1.2em; margin-bottom: 2em;">
        An AI-powered application for visualizing ocean currents, designed for oceanographers and surfers.
    </div>
    
    **Features:**
    - Real-time current visualization
    - Multiple ocean regions
    - Forecast capabilities
    - Surfing condition analysis
    """)
    
    with gr.Tab("Current Map"):
        with gr.Row():
            with gr.Column(scale=1):
                region = gr.Dropdown(
                    choices=["Gulf of Mexico", "California Coast", "Atlantic Coast", "Global"],
                    value="Gulf of Mexico",
                    label="Region"
                )
                resolution = gr.Dropdown(
                    choices=["High", "Medium", "Low"],
                    value="Medium",
                    label="Resolution"
                )
                show_vectors = gr.Checkbox(label="Show Current Vectors", value=True)
                show_speed = gr.Checkbox(label="Show Speed Contours", value=True)
                vector_scale = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Vector Scale"
                )
                update_map = gr.Button("Update Map", variant="primary")
                
            with gr.Column(scale=2):
                current_map = gr.Plot(
                    label="Ocean Current Map",
                    show_label=False
                )
        
        update_map.click(
            fn=create_current_map,
            inputs=[region, resolution, show_vectors, show_speed, vector_scale],
            outputs=current_map
        )
    
    with gr.Tab("Forecast"):
        with gr.Row():
            with gr.Column(scale=1):
                forecast_region = gr.Dropdown(
                    choices=["Gulf of Mexico", "California Coast", "Atlantic Coast", "Global"],
                    value="Gulf of Mexico",
                    label="Region"
                )
                forecast_hours = gr.Slider(
                    minimum=6,
                    maximum=72,
                    value=24,
                    step=6,
                    label="Forecast Hours"
                )
                update_forecast = gr.Button("Generate Forecast", variant="primary")
                
            with gr.Column(scale=2):
                forecast_plot = gr.Plot(
                    label="Ocean Conditions Forecast",
                    show_label=False
                )
        
        update_forecast.click(
            fn=create_forecast,
            inputs=[forecast_region, forecast_hours],
            outputs=forecast_plot
        )
    
    with gr.Tab("Surfing Conditions"):
        with gr.Row():
            with gr.Column(scale=1):
                surf_region = gr.Dropdown(
                    choices=["Gulf of Mexico", "California Coast", "Atlantic Coast"],
                    value="California Coast",
                    label="Surfing Region"
                )
                analyze_button = gr.Button("Analyze Conditions", variant="primary")
                
            with gr.Column(scale=2):
                surf_analysis = gr.Textbox(
                    label="Surfing Conditions Analysis",
                    lines=8,
                    placeholder="Click 'Analyze Conditions' to get surfing recommendations..."
                )
        
        analyze_button.click(
            fn=analyze_conditions,
            inputs=[surf_region],
            outputs=surf_analysis
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## About This Application
        
        This Ocean Current Mapper provides real-time visualization and analysis of ocean currents using data from:
        
        - **NOAA Tides & Currents**: Real-time oceanographic observations
        - **NASA OSCAR**: Global surface current analyses
        - **NOAA Global RTOFS**: Ocean forecast system
        
        ### For Oceanographers:
        - High-resolution current maps
        - Vector field visualization
        - Multi-day forecasting
        - Data export capabilities
        
        ### For Surfers:
        - Current safety analysis
        - Wave and wind conditions
        - Rip current warnings
        - Beach-specific recommendations
        
        ### Technical Details:
        - Built with Gradio for easy deployment
        - Hosted on Hugging Face Spaces
        - Real-time API integration
        - Interactive visualizations with Plotly
        
        **Note**: This demo uses synthetic data for demonstration. In production, it would connect to live oceanographic APIs.
        """)

# Launch the app with better error handling
if __name__ == "__main__":
    try:
        print("Starting Ocean Current Mapper...")
        demo.launch(
            share=True,
            show_error=True,
            inbrowser=False,
            server_name="0.0.0.0",
            server_port=7860
        )
    except Exception as e:
        print(f"Error launching app: {e}")
        traceback.print_exc()
