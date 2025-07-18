import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
import json
import re
import asyncio
import os
from datetime import datetime
import warnings
import requests
import math
from agents import Agent, function_tool, Runner
from openai.types.responses import ResponseTextDeltaEvent

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RHR Multi-Agent Property AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .agent-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .success-box {
        background-color: #efe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .cross-agent-indicator {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Agent configurations with visualization settings
AGENT_CONFIGS = {
    'condo': {
        'name': 'Condo Expert',
        'icon': '🏠',
        'table': 'condo_converted_2025',
        'color': '#3498db',
        'description': 'Residential condominium specialist',
        'visualization': {
            'primary_name': 'project_name',
            'tooltip_columns': ['project_name', 'developer', 'wadmpr', 'wadmkk', 'grade', 'unit'],
            'column_labels': ['Project', 'Developer', 'Provinsi', 'Kab/Kota', 'Grade', 'Units'],
            'format_rules': {'unit': 'number'},
            'fallback_columns': ['id', 'address']
        }
    },
    'hotel': {
        'name': 'Hotel Expert', 
        'icon': '🏨',
        'table': 'hotel_converted_2025',
        'color': '#e74c3c',
        'description': 'Hospitality property specialist',
        'visualization': {
            'primary_name': 'project_name',
            'tooltip_columns': ['project_name', 'management', 'wadmpr', 'wadmkk', 'star', 'unit_developed'],
            'column_labels': ['Project', 'Management', 'Provinsi', 'Kab/Kota', 'Star', 'Units'],
            'format_rules': {'unit_developed': 'number'},
            'fallback_columns': ['id', 'address']
        }
    },
    'hospital': {
        'name': 'Hospital Expert',
        'icon': '🏥',
        'table': 'hospital_converted_2025',
        'color': '#9b59b6',
        'description': 'Healthcare facility specialist',
        'visualization': {
            'primary_name': 'object_name',
            'tooltip_columns': ['object_name', 'type', 'wadmpr', 'wadmkk', 'grade', 'beds_capacity'],
            'column_labels': ['Hospital', 'Type', 'Provinsi', 'Kab/Kota', 'Grade', 'Beds'],
            'format_rules': {'beds_capacity': 'number'},
            'fallback_columns': ['id']
        }
    },
    'office': {
        'name': 'Office Expert',
        'icon': '🏢',
        'table': 'office_converted_2025',
        'color': '#f39c12',
        'description': 'Commercial office specialist',
        'visualization': {
            'primary_name': 'building_name',
            'tooltip_columns': ['building_name', 'owner/developer', 'wadmpr', 'wadmkk', 'grade', 'price_avg'],
            'column_labels': ['Building', 'Owner/Developer', 'Provinsi', 'Kab/Kota', 'Grade', 'Price Avg'],
            'format_rules': {'price_avg': 'currency'},
            'fallback_columns': ['id']
        }
    },
    'retail': {
        'name': 'Retail Expert',
        'icon': '🏬',
        'table': 'retail_converted_2025',
        'color': '#27ae60',
        'description': 'Retail property specialist',
        'visualization': {
            'primary_name': 'project_name',
            'tooltip_columns': ['project_name', 'developer', 'wadmpr', 'wadmkk', 'grade', 'price_avg'],
            'column_labels': ['Project', 'Developer', 'Provinsi', 'Kab/Kota', 'Grade', 'Price Avg'],
            'format_rules': {'price_avg': 'currency'},
            'fallback_columns': ['id', 'address']
        }
    },
    'land': {
        'name': 'Land Market Expert',
        'icon': '🌍',
        'table': 'engineered_property_data',
        'color': '#8b4513',
        'description': 'Land market and property value specialist',
        'visualization': {
            'primary_name': 'alamat',
            'tooltip_columns': ['alamat', 'wadmpr', 'wadmkk', 'wadmkc', 'hpm', 'luas_tanah'],
            'column_labels': ['Alamat', 'Provinsi', 'Kab/Kota', 'Kecamatan', 'HPM', 'Luas Tanah'],
            'format_rules': {'hpm': 'currency_per_m2', 'luas_tanah': 'area'},
            'fallback_columns': ['id']
        }
    }
}

class DatabaseConnection:
    """Handle PostgreSQL database connections"""
    
    def __init__(self):
        self.engine = None
        self.connection_status = False
    
    def connect(self, db_user: str, db_pass: str, db_host: str, 
                db_port: str, db_name: str, schema: str = 'public'):
        """Establish database connection"""
        try:
            connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.connection_status = True
            return True, "Connection successful!"
        
        except Exception as e:
            self.connection_status = False
            return False, f"Connection failed: {str(e)}"
    
    def execute_query(self, query: str) -> tuple:
        """Execute SQL query and return results"""
        try:
            if not self.connection_status:
                return None, "No database connection established"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = []
                columns = list(result.keys())
                
                for row in result:
                    row_dict = {}
                    for i, value in enumerate(row):
                        row_dict[columns[i]] = value
                    rows.append(row_dict)
                
                if rows:
                    df = pd.DataFrame(rows)
                else:
                    df = pd.DataFrame(columns=columns)
                
                return df, "Query executed successfully"
        
        except Exception as e:
            return None, f"Query execution failed: {str(e)}"

class GeocodeService:
    """Handle geocoding using Google Maps Geocoding API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def geocode_address(self, address: str) -> tuple:
        """Geocode an address to get latitude and longitude"""
        try:
            params = {
                'address': address,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                result = data['results'][0]
                location = result['geometry']['location']
                formatted_address = result['formatted_address']
                
                return (
                    location['lat'],
                    location['lng'],
                    formatted_address
                )
            else:
                return None, None, None
                
        except Exception as e:
            st.error(f"Geocoding error: {str(e)}")
            return None, None, None

# Adaptive visualization helper functions
class AdaptiveVisualizationHelper:
    """Handle adaptive visualization for different agent types and table structures"""
    
    @staticmethod
    def get_agent_config(agent_type: str = None) -> dict:
        """Get current agent's visualization configuration"""
        if agent_type is None:
            agent_type = st.session_state.get('current_agent', 'condo')
        
        return AGENT_CONFIGS.get(agent_type, {}).get('visualization', {})
    
    @staticmethod
    def detect_available_columns(df, config: dict) -> tuple:
        """Detect available columns and create fallback mapping"""
        tooltip_columns = config.get('tooltip_columns', [])
        column_labels = config.get('column_labels', [])
        fallback_columns = config.get('fallback_columns', ['id'])
        
        # Find available columns
        available_columns = []
        available_labels = []
        
        for i, col in enumerate(tooltip_columns):
            if col in df.columns:
                available_columns.append(col)
                available_labels.append(column_labels[i] if i < len(column_labels) else col)
            else:
                # Try to find fallback
                for fallback in fallback_columns:
                    if fallback in df.columns:
                        available_columns.append(fallback)
                        available_labels.append(fallback.replace('_', ' ').title())
                        break
        
        # Always add distance_km if available
        if 'distance_km' in df.columns and 'distance_km' not in available_columns:
            available_columns.append('distance_km')
            available_labels.append('Distance')
        
        return available_columns, available_labels
    
    @staticmethod
    def format_value(value, format_type: str) -> str:
        """Format value based on format type"""
        if pd.isna(value) or value == '' or value is None:
            return '-'
        
        try:
            if format_type == 'currency':
                return f"Rp {float(value):,.0f}"
            elif format_type == 'currency_per_m2':
                return f"Rp {float(value):,.0f}/m²"
            elif format_type == 'area':
                return f"{float(value):,.0f} m²"
            elif format_type == 'number':
                return f"{float(value):,.0f}"
            elif format_type == 'distance':
                return f"{float(value):.2f} km"
            else:
                return str(value)
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def create_adaptive_tooltip(df, agent_type: str = None) -> tuple:
        """Create adaptive tooltip data and template"""
        config = AdaptiveVisualizationHelper.get_agent_config(agent_type)
        available_columns, available_labels = AdaptiveVisualizationHelper.detect_available_columns(df, config)
        format_rules = config.get('format_rules', {})
        
        if not available_columns:
            # Fallback to basic columns
            available_columns = ['id']
            available_labels = ['ID']
        
        # Ensure all columns exist in dataframe
        for col in available_columns:
            if col not in df.columns:
                if col == 'distance_km':
                    df[col] = 0.0
                else:
                    df[col] = ''
        
        # Build hover template
        hover_parts = []
        for i, (col, label) in enumerate(zip(available_columns, available_labels)):
            format_type = format_rules.get(col, 'default')
            
            if format_type == 'currency':
                hover_parts.append(f"{label}: Rp %{{customdata[{i}]:,.0f}}")
            elif format_type == 'currency_per_m2':
                hover_parts.append(f"{label}: Rp %{{customdata[{i}]:,.0f}}/m²")
            elif format_type == 'area':
                hover_parts.append(f"{label}: %{{customdata[{i}]:,.0f}} m²")
            elif format_type == 'number':
                hover_parts.append(f"{label}: %{{customdata[{i}]:,.0f}}")
            elif col == 'distance_km':
                hover_parts.append(f"{label}: %{{customdata[{i}]:.2f}} km")
            else:
                hover_parts.append(f"{label}: %{{customdata[{i}]}}")
        
        hover_template = "<br>".join(hover_parts) + "<extra></extra>"
        
        # Create customdata array
        customdata = df[available_columns].fillna('').values
        
        return customdata, hover_template, available_columns

# Function tools for agents
@function_tool
def execute_sql_query(sql_query: str) -> str:
    """Execute SQL query and return formatted results"""
    try:
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            # Store for future reference WITH the query
            st.session_state.last_query_result = result_df.copy()
            st.session_state.last_executed_query = sql_query  # Store the actual query
            st.session_state.last_query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Display results in expandable section
            with st.expander("📊 Query Results", expanded=False):
                st.code(sql_query, language="sql")
                st.dataframe(result_df, use_container_width=True)
            
            # Return formatted summary
            if len(result_df) == 1 and len(result_df.columns) == 1:
                # Single value result (like COUNT)
                value = result_df.iloc[0, 0]
                return f"Query result: {value}"
            elif len(result_df) <= 10:
                # Small result - show all
                return f"Query returned {len(result_df)} rows:\n{result_df.to_string(index=False)}"
            else:
                # Large result - show summary
                return f"Query returned {len(result_df)} rows. Data displayed in expandable section above."
        else:
            return f"No results returned: {query_msg}"
            
    except Exception as e:
        return f"SQL Error: {str(e)}"

@function_tool
def create_map_visualization(sql_query: str, title: str = "Property Locations") -> str:
    """Create a map visualization from SQL query results"""
    try:
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is None or len(result_df) == 0:
            return f"Error: No data returned from query - {query_msg}"
        
        # Check required columns
        if 'latitude' not in result_df.columns or 'longitude' not in result_df.columns:
            return "Error: Query must include 'latitude' and 'longitude' columns for map visualization"
        
        # Clean and filter coordinates
        map_df = result_df.copy()
        map_df = map_df.dropna(subset=['latitude', 'longitude'])
        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
        
        # Filter valid coordinates
        map_df = map_df[
            (map_df['latitude'] >= -90) & (map_df['latitude'] <= 90) &
            (map_df['longitude'] >= -180) & (map_df['longitude'] <= 180) &
            (map_df['latitude'] != 0) & (map_df['longitude'] != 0)
        ]
        
        if len(map_df) == 0:
            return "Error: No valid coordinates found in the data"
        
        # Use adaptive tooltip system
        customdata, hover_template, tooltip_columns = AdaptiveVisualizationHelper.create_adaptive_tooltip(map_df)
        
        # Add markers with adaptive tooltip
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=map_df['latitude'],
            lon=map_df['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            customdata=customdata,
            hovertemplate=hover_template,
            name=f'Properties ({len(map_df)})'
        ))
        
        # Calculate center
        center_lat = map_df['latitude'].mean()
        center_lon = map_df['longitude'].mean()
        
        # Map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=8
            ),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            title=title
        )
        
        # Store the figure in session state for persistence
        st.session_state.last_visualization = {
            "type": "map",
            "figure": fig,
            "title": title
        }

        # Display map
        st.plotly_chart(fig, use_container_width=True)
        
        # Store for future reference
        st.session_state.last_query_result = map_df.copy()
        st.session_state.last_map_data = map_df.copy()
        
        st.session_state.last_executed_query = sql_query  # Store the actual query
        st.session_state.last_query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_visualization_type = "map"
        
        # Show query details
        with st.expander("🗺️ Map Query Details", expanded=False):
            st.code(sql_query, language="sql")
            st.info(f"Mapped {len(map_df)} properties with valid coordinates")
        
        return f"✅ Map successfully created with {len(map_df)} properties"
        
    except Exception as e:
        return f"Error creating map: {str(e)}"

@function_tool
def create_chart_visualization(chart_type: str, sql_query: str, title: str, 
                              x_column: str = None, y_column: str = None, 
                              color_column: str = None) -> str:
    """Create chart visualizations with smart color handling for line charts"""
    try:
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is None or len(result_df) == 0:
            return f"Error: No data returned from query - {query_msg}"
        
        # Generate unique key for chart
        import hashlib
        import time
        unique_key = hashlib.md5(f"{sql_query}_{title}_{time.time()}".encode()).hexdigest()[:8]
        
        # Auto-detect columns if not provided
        if x_column is None or x_column not in result_df.columns:
            x_column = result_df.columns[0]
        if y_column is None or y_column not in result_df.columns:
            numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
            y_column = numeric_cols[0] if numeric_cols else result_df.columns[1] if len(result_df.columns) > 1 else None
        
        fig = None
        
        # Create chart based on type
        if chart_type == "line":
            # FIX: Smart color column handling for line charts
            line_df = result_df.copy()
            
            # Sort by x-axis
            line_df = line_df.sort_values(x_column)
            
            # Check if we should use color column for line charts
            use_color = None
            if color_column and color_column in line_df.columns:
                # Count unique values per color group
                color_counts = line_df.groupby(color_column).size()
                
                # Only use color if multiple points per group (for proper lines)
                if color_counts.min() >= 2:
                    use_color = color_column
                # If single points per color, don't use color (creates disconnected points)
                else:
                    use_color = None
                    st.info(f"Removed color grouping for line chart - each category has only 1 point")
            
            # Create line chart with conditional color
            fig = px.line(line_df, x=x_column, y=y_column, color=use_color, 
                         title=title, markers=True)
            
            # Make lines more visible
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            
        elif chart_type == "bar":
            fig = px.bar(result_df, x=x_column, y=y_column, color=color_column, title=title)
            fig.update_layout(xaxis_tickangle=-45)
            
        elif chart_type == "pie":
            if y_column:
                fig = px.pie(result_df, names=x_column, values=y_column, title=title)
            else:
                pie_data = result_df[x_column].value_counts().reset_index()
                pie_data.columns = [x_column, 'count']
                fig = px.pie(pie_data, names=x_column, values='count', title=title)
                
        elif chart_type == "scatter":
            fig = px.scatter(result_df, x=x_column, y=y_column, color=color_column, title=title)
            
        elif chart_type == "histogram":
            fig = px.histogram(result_df, x=x_column if x_column else y_column, color=color_column, title=title)
            
        else:
            # Default to bar chart
            fig = px.bar(result_df, x=x_column, y=y_column, color=color_column, title=title)
            fig.update_layout(xaxis_tickangle=-45)
        
        if fig:
            fig.update_layout(
                height=500,
                template="plotly_white",
                title_x=0.5,
                margin=dict(l=50, r=50, t=80, b=100)
            )
        
        if fig:
            # Store the figure in session state for persistence
            st.session_state.last_visualization = {
                "type": "chart",
                "figure": fig,
                "chart_type": chart_type,
                "title": title,
                "unique_key": unique_key
            }
            
            # Display chart with unique key
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_key}")
            
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            st.session_state.last_executed_query = sql_query
            st.session_state.last_query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.last_visualization_type = "chart"
            
            # Show query details
            with st.expander("📊 Chart Query Details", expanded=False):
                st.code(sql_query, language="sql")
                st.dataframe(result_df, use_container_width=True)
            
            return f"✅ {chart_type.title()} chart successfully created with {len(result_df)} data points"
        else:
            return "Error: Failed to create chart"
            
    except Exception as e:
        return f"Error creating chart: {str(e)}"

@function_tool
def find_nearby_projects(location_name: str, radius_km: float = 1.0, 
                        title: str = None) -> str:
    """Find and map projects near a specific location using geocoding"""
    try:
        if not hasattr(st.session_state, 'geocode_service') or st.session_state.geocode_service is None:
            return "Error: Geocoding service not available. Please add Google Maps API key."
        
        # Set default title
        if title is None:
            title = f"Projects within {radius_km} km from {location_name}"
        
        # Geocode the location
        lat, lng, formatted_address = st.session_state.geocode_service.geocode_address(location_name)
        
        if lat is None or lng is None:
            return f"Error: Could not find coordinates for location '{location_name}'. Try being more specific."
        
        st.success(f"📍 Location found: {formatted_address}")
        st.info(f"Coordinates: {lat:.6f}, {lng:.6f}")
        
        # Use the table from the current active agent
        current_agent_type = st.session_state.current_agent
        table_name = AGENT_CONFIGS[current_agent_type]['table']
        
        # Build query based on agent type
        if current_agent_type == 'land':
            # For land agent - use engineered_property_data table
            sql_query = f"""
                SELECT 
                    id,
                    alamat,
                    CAST(latitude AS NUMERIC) as latitude,
                    CAST(longitude AS NUMERIC) as longitude,
                    wadmpr,
                    wadmkk,
                    wadmkc,
                    wadmkd,
                    hpm,
                    luas_tanah,
                    ST_DistanceSphere(
                        ST_MakePoint(CAST(longitude AS NUMERIC), CAST(latitude AS NUMERIC)),
                        ST_MakePoint({lng}, {lat})
                    ) / 1000 AS distance_km
                FROM {table_name}
                WHERE
                    latitude IS NOT NULL 
                    AND longitude IS NOT NULL
                    AND latitude != '' 
                    AND longitude != ''
                    AND latitude NOT LIKE '%null%'
                    AND longitude NOT LIKE '%null%'
                    AND ST_DWithin(
                        ST_MakePoint(CAST(longitude AS NUMERIC), CAST(latitude AS NUMERIC))::geography,
                        ST_MakePoint({lng}, {lat})::geography,
                        {radius_km} * 1000
                    )
                ORDER BY distance_km ASC
                LIMIT 100
            """
            display_columns = ['alamat', 'wadmpr', 'wadmkk', 'wadmkc', 'hpm', 'luas_tanah', 'distance_km']
            
        else:
            # For other agents - use their respective tables
            if current_agent_type == 'condo':
                main_column = 'project_name'
                additional_columns = 'address, developer, grade, unit'
            elif current_agent_type == 'hotel':
                main_column = 'project_name'
                additional_columns = 'address, star, management, unit_developed'
            elif current_agent_type == 'office':
                main_column = 'building_name'
                additional_columns = 'grade, "owner/developer", price_avg'
            elif current_agent_type == 'hospital':
                main_column = 'object_name'
                additional_columns = 'type, grade, beds_capacity'
            elif current_agent_type == 'retail':
                main_column = 'project_name'
                additional_columns = 'address, developer, grade, price_avg'
            else:
                main_column = 'id'
                additional_columns = ''
            
            sql_query = f"""
                SELECT 
                    id,
                    {main_column},
                    latitude,
                    longitude,
                    {additional_columns},
                    wadmpr,
                    wadmkk,
                    wadmkc,
                    ST_DistanceSphere(
                        ST_MakePoint(longitude, latitude),
                        ST_MakePoint({lng}, {lat})
                    ) / 1000 AS distance_km
                FROM {table_name}
                WHERE
                    latitude IS NOT NULL 
                    AND longitude IS NOT NULL
                    AND latitude != 0 
                    AND longitude != 0
                    AND ST_DWithin(
                        ST_MakePoint(longitude, latitude)::geography,
                        ST_MakePoint({lng}, {lat})::geography,
                        {radius_km} * 1000
                    )
                ORDER BY distance_km ASC
                LIMIT 100
            """
            display_columns = [main_column, 'wadmpr', 'wadmkk', 'distance_km']
        
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            # Create enhanced map with reference point
            fig = go.Figure()
            
            # Add reference point (target location)
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lng],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='star'),
                text=[f"📍 {location_name}<br>{formatted_address}"],
                hovertemplate='%{text}<extra></extra>',
                name='Target Location'
            ))
            
            # Use adaptive tooltip system for project markers
            customdata, hover_template, tooltip_columns = AdaptiveVisualizationHelper.create_adaptive_tooltip(result_df, current_agent_type)
            
            # Use agent-specific color
            marker_color = AGENT_CONFIGS[current_agent_type]['color']
            
            fig.add_trace(go.Scattermapbox(
                lat=result_df['latitude'],
                lon=result_df['longitude'],
                mode='markers',
                marker=dict(size=8, color=marker_color),
                customdata=customdata,
                hovertemplate=hover_template,
                name=f'{current_agent_type.title()} Properties ({len(result_df)})'
            ))
            
            # Map layout centered on target location
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=lat, lon=lng),
                    zoom=12
                ),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                title=title
            )
            
            # Store the figure in session state for persistence
            st.session_state.last_visualization = {
                "type": "nearby_map",
                "figure": fig,
                "title": title,
                "location": location_name,
                "radius": radius_km,
                "count": len(result_df)
            }

            # Display map
            st.plotly_chart(fig, use_container_width=True)

            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            st.session_state.last_map_data = result_df.copy()
            
            st.session_state.last_executed_query = sql_query
            st.session_state.last_query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.last_visualization_type = "nearby_search"
            
            # Show results table
            with st.expander("📊 Nearby Projects Details", expanded=False):
                st.code(sql_query, language="sql")
                if display_columns:
                    available_cols = [col for col in display_columns if col in result_df.columns]
                    if available_cols:
                        st.dataframe(result_df[available_cols].round(2), use_container_width=True)
                    else:
                        st.dataframe(result_df.round(2), use_container_width=True)
                else:
                    st.dataframe(result_df.round(2), use_container_width=True)

            return f"✅ Found {len(result_df)} {current_agent_type} properties within {radius_km} km from {location_name}. Closest property is {result_df['distance_km'].min():.2f} km away."
        
        else:
            return f"❌ No {current_agent_type} properties found within {radius_km} km from {location_name}. Query message: {query_msg}"
        
    except Exception as e:
        return f"Error finding nearby projects: {str(e)}"


def get_agent_instructions(agent_type: str, table_name: str) -> str:
    """Get agent-specific instructions with table name"""
    
    instructions = {
        'condo': f"""You are a Condominium Property Expert AI for a public appraisal services office in Indonesia specializing in residential condominiums.
Table: {table_name}

CONDO EXPERTISE:
- Residential condominium analysis
- Developer performance and delivery
- Unit counts and residential capacity
- Condo grades and market positioning

COLUMN INFORMATION:
- id (INTEGER)
- geometry (TEXT): Geospatial geometry field (usually for PostGIS spatial data).
- latitude, longitude (DOUBLE PRECISION): Official/project-recorded latitude and longitude coordinates.
- completionyear (INTEGER)
- q (INTEGER)
- project_status (TEXT)
- project_name (TEXT)
- address (TEXT)
- developer (TEXT)
- area (TEXT)
- precinct (TEXT)
- grade (TEXT)
- unit (INTEGER)
- wadmpr (TEXT): Province (e.g., "DKI Jakarta").
- wadmkk (TEXT): Regency/City (e.g., "Jakarta Selatan").
- wadmkc (TEXT): District (e.g., "Tebet").

SQL RULES:
- unit is INTEGER - use directly: SUM(unit), AVG(unit)
- NO PRICING DATA available in this table
- For maps: Include id, latitude, longitude, project_name, address, grade

RESPONSE STYLE:
- Always respond in user's language (auto-detect)
- Provide residential market insights with data
- Use tools appropriately based on request type
- Handle follow-up questions using context

CRITICAL: You can ONLY answer questions in this condo property domain scope!""",

        'hotel': f"""You are a Hotel Property Expert AI for a public appraisal services office in Indonesia specializing in hospitality properties.
Table: {table_name}

HOTEL EXPERTISE:
- Hotel and hospitality analysis
- Star rating classifications (1-5 stars)
- Hotel management and operations
- Event facilities and capacity

COLUMN DETAILS:
- id (INTEGER)
- geometry (TEXT)
- latitude/longitude (DOUBLE PRECISION)
- completionyear (INTEGER)
- q (INTEGER)
- project_status (TEXT)
- project_name (TEXT)
- address (TEXT)
- developer (TEXT)
- management (TEXT)
- area (TEXT)
- precinct (TEXT)
- star (TEXT)
- concept (TEXT)
- unit_developed (INTEGER)
- ballroom_capacity (INTEGER)
- price_2016 to price_2025 (TEXT)
- price_avg (TEXT)
- wadmpr (TEXT)
- wadmkk (TEXT)
- wadmkc (TEXT)

SQL RULES:
- unit_developed, ballroom_capacity are INTEGER - use directly
- Price columns are TEXT - use CAST(price_avg AS NUMERIC)
- For maps: Include id, latitude, longitude, project_name, star, concept
- Always include LIMIT to prevent large results

RESPONSE STYLE:
- Always respond in user's language (auto-detect)
- Provide hospitality insights with data
- Use tools appropriately based on request type
- Handle follow-up questions using context

CRITICAL: You can ONLY answer questions in this hotel property domain scope!""",

        'office': f"""You are an Office Property Expert AI for a public appraisal services office in Indonesia specializing in commercial office spaces.
Table: {table_name}

OFFICE EXPERTISE:
- Commercial office building analysis
- Grade A, B, C office classifications
- Office rental rates and pricing trends
- Corporate real estate analysis

COLUMN DETAILS:
- id (INTEGER), geometry (TEXT), latitude/longitude (DOUBLE PRECISION)
- building_name (TEXT), grade (TEXT), project_status (TEXT)
- sga (TEXT), gfa (TEXT), "owner/developer" (TEXT)
- price_2016 to price_2025 (DOUBLE PRECISION), price_avg (DOUBLE PRECISION)
- wadmpr (TEXT), wadmkk (TEXT), wadmkc (TEXT)

SQL RULES:
- Price columns are DOUBLE PRECISION - use directly: AVG(price_avg)
- Column "owner/developer" needs quotes
- For maps: Include id, latitude, longitude, building_name, grade, price_avg
- Always include LIMIT to prevent large results

RESPONSE STYLE:
- Always respond in user's language (auto-detect)
- Provide office market insights with data
- Use tools appropriately based on request type
- Handle follow-up questions using context

CRITICAL: You can ONLY answer questions in this office property domain scope!""",

        'hospital': f"""You are a Hospital Property Expert AI for a public appraisal services office in Indonesia specializing in healthcare facilities using.
Table: {table_name}

HOSPITAL EXPERTISE:
- Healthcare facility analysis and capacity
- Medical services and BPJS coverage
- Hospital grades and ownership types
- Healthcare accessibility analysis

COLUMN DETAILS:
- id (INTEGER), geometry (TEXT), latitude/longitude (DOUBLE PRECISION)
- object_name (TEXT), type (TEXT), grade (TEXT), ownership (TEXT)
- beds_capacity (INTEGER), land_area (TEXT), building_area (TEXT)
- bpjs (TEXT), kb_gratis (TEXT)
- wadmpr (TEXT), wadmkk (TEXT), wadmkc (TEXT)

SQL RULES:
- beds_capacity is INTEGER - use directly for SUM(), AVG()
- land_area/building_area are TEXT - use CAST(land_area AS NUMERIC)
- For maps: Include id, latitude, longitude, object_name, type, grade
- Always include LIMIT to prevent large results

RESPONSE STYLE:
- Always respond in user's language (auto-detect)
- Provide healthcare insights with data
- Use tools appropriately based on request type
- Handle follow-up questions using context

CRITICAL: You can ONLY answer questions in this hospital property domain scope!""",

        'retail': f"""You are a Retail Property Expert AI for a public appraisal services office in Indonesia specializing in commercial retail spaces using.
Table: {table_name}

RETAIL EXPERTISE:
- Shopping malls, retail outlets, commercial spaces
- Net Lettable Area (NLA) and Gross Floor Area (GFA) analysis
- Retail pricing trends and market analysis
- Developer and project performance

COLUMN DETAILS:
- id (INTEGER)
- geometry (TEXT)
- latitude/longitude (DOUBLE PRECISION)
- project_name (TEXT)
- address (TEXT)
- developer (TEXT)
- grade (TEXT)
- nla (TEXT)
- gfa (TEXT)
- price_2016 to price_2025 (TEXT)
- price_avg (TEXT)
- wadmpr (TEXT)
- wadmkk (TEXT)
- wadmkc (TEXT)

SQL RULES:
- Price columns are TEXT - use CAST(price_avg AS NUMERIC) for calculations
- nla and gfa are TEXT - use CAST(nla AS NUMERIC) if needed
- For maps: Include id, latitude, longitude, project_name, address, grade
- Always include LIMIT to prevent large results

RESPONSE STYLE:
- Always respond in user's language (auto-detect)
- Provide retail market insights with data
- Use tools appropriately based on request type
- Handle follow-up questions using context

CRITICAL: You can ONLY answer questions in this retail property domain scope!""",

        'land': f"""You are a Land Market Expert for a public appraisal services office in Indonesia specializing in land property analysis using.
Table: {table_name}

LAND EXPERTISE:
- Land valuation and price per meter analysis
- Land characteristics and development potential
- Geographic market trends and accessibility
- Area-based pricing comparison

COLUMN DETAILS:
- id (INTEGER)
- alamat (TEXT): Property address.
- latitude/longitude (TEXT): Official/project-recorded latitude and longitude coordinates.
- jenis_objek (INTEGER): Property Object Type in number (e.g., 1,2, ect.).
- luas_tanah (FLOAT): Property area in squared meter.
- bentuk_tapak (TEXT): Shape of the property site (Letter L, Persegi Panjang, Kipas, Persegi, Trapesium, Tidak Beraturan, Ngantong, Menggantung).
- posisi_tapak (TEXT): Property site position (Diapit Jalan, Helikopter, Sudut/Hook, Buntu, Tengah, Lainnya, Tusuk Sate, Ujung, Hook, Pojok, Tusuk Sata (Tusuk sate and Tusuk Sata have the same meaning, so just SUM it)).
- orientasi (TEXT): Property orientation (Selatan, Barat Daya, Timur Laut, Barat, Utara, Timur, Tenggara, Barat Laut).
- lebar_jalan_di_depan (FLOAT): Wide road ahead the property in meter.
- kondisi_wilayah_sekitar (TEXT): Condition of the area around the property (Perumahan Menengah, Industri, Industri/Pergudangan Besar, Campuran, Industri / Perdagangan UKM, Rawan Bencana, Perumahan Sederhana, Perumahan Mewah, Komersial UKM, Kosong Pertanian, Lainnya, Pemerintahan, Dekat Sungai / Parit, Hijau, Dekat TPU, Komersial Primer, Komersial Menengah, Industri / Perdagangan Besar, Komersial, Perumahan).
- jenis_jalan_utama (TEXT): Type of main road (Jalan, Gang)
- perkerasan_jalan (TEXT): Property road paving (Sirtu, Lainnya, Aspal Hotmix, Tanah, Aspal Siram, Beton, Aspal, Aspak Hotmix, Paving)
- hpm (FLOAT) : Property Price per squared meter.
- tahun_pengambilan_data (INTEGER) : Year per date of the property price being taken or being survey (not the year of the property build).
- wadmpr (TEXT): Province (e.g., "DKI Jakarta").
- wadmkk (TEXT): Regency/City (e.g., "Jakarta Selatan").
- wadmkc (TEXT): District (e.g., "Tebet").
- wadmkd (TEXT): Subdistrict/village (e.g., "Manggarai").

SQL RULES:
- Always include LIMIT if user want to show the data in table to prevent large results, if not just calculate or analyze.

RESPONSE STYLE:
- Always respond in user's language (auto-detect)
- Provide land market insights with data
- Use tools appropriately based on request type
- Handle follow-up questions using context

CRITICAL: You can ONLY answer questions in this land property domain scope! """
    }
    
    return instructions.get(agent_type, f"You are a {agent_type} property expert using table {table_name}")

def initialize_agents():
    """Initialize all property agents using gpt-4.1"""
    
    # Set OpenAI API key
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
    except KeyError:
        st.error("OpenAI API key not found in secrets.toml")
        return None
    
    agents = {}
    
    # Create specialized agents for each property type
    for agent_type, config in AGENT_CONFIGS.items():
        table_name = config['table']
        
        # Get agent-specific instructions
        agent_instructions = get_agent_instructions(agent_type, table_name)
        
        # Add common tool instructions
        full_instructions = f"""{agent_instructions}

**TOOLS:**
1. execute_sql_query(sql) - Run queries, show data
2. create_map_visualization(sql, title) - Auto-map for coordinates  
3. create_chart_visualization(type, sql, title, x, y) - Charts for trends
4. find_nearby_projects(location, radius) - Geocoded proximity search

**RESPONSE:** 
- General Questions : Detect intent → general answer in user's language.
- General Questions + Gain info from database : Detect intent → ask user for more spesific instruction or select columns → query → execute → show results + general answer in user's language.
- General Questions + Gain info from database + With tools : Detect intent → ask user for more spesific instruction or select columns → query → execute → show results + general answer in user's language → detect intent → select tools → execute → show results + general answer in user's language.
"""
        
        # Create agent
        agent = Agent(
            name=f"{agent_type}_expert",
            instructions=full_instructions,
            model="gpt-4.1",
            tools=[
                execute_sql_query,
                create_map_visualization,
                create_chart_visualization,
                find_nearby_projects
            ]
        )
        
        agents[agent_type] = agent
    
    return agents

# Cross-agent query parser
class CrossAgentQueryParser:
    def __init__(self):
        self.patterns = {
            'comparison': r'#(\w+)(\s+vs\s+\w+)+',
            'consultation': r'#(\w+)\s+consult\s+([\w\s]+)',
            'market_analysis': r'#all\s+(.+)',
            'impact': r'#(\w+)\s+impact\s+(\w+)'
        }
    
    def parse_query(self, question: str) -> dict:
        if not question.startswith('#'):
            return {'type': 'single_agent', 'agents': []}
        
        for query_type, pattern in self.patterns.items():
            match = re.match(pattern, question.lower())
            if match:
                return self._extract_agents(query_type, match, question)
        
        return {'type': 'invalid', 'error': 'Invalid cross-agent syntax'}
    
    def _extract_agents(self, query_type: str, match, original_query: str) -> dict:
        if query_type == 'comparison':
            agents_text = match.group(0)[1:]  # Remove #
            agents = re.findall(r'(\w+)', agents_text)
            valid_agents = [a for a in agents if a in AGENT_CONFIGS]
            return {
                'type': 'comparison',
                'agents': valid_agents,
                'primary': valid_agents[0] if valid_agents else None,
                'original_query': original_query
            }
        # Add other query types as needed
        return {'type': 'invalid', 'error': 'Could not parse query'}

# Authentication functions
def check_authentication():
    return st.session_state.get('authenticated', False)

def login():
    st.markdown('<div class="section-header">🔐 Login</div>', unsafe_allow_html=True)
    
    try:
        valid_username = st.secrets["auth"]["username"]
        valid_password = st.secrets["auth"]["password"]
    except KeyError:
        st.error("Authentication credentials not found in secrets.toml")
        return False
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username == valid_username and password == valid_password:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    return False

# Database initialization
def initialize_database():
    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = DatabaseConnection()
    
    if not st.session_state.db_connection.connection_status:
        try:
            db_user = st.secrets["database"]["user"]
            db_pass = st.secrets["database"]["password"]
            db_host = st.secrets["database"]["host"]
            db_port = st.secrets["database"]["port"]
            db_name = st.secrets["database"]["name"]
            schema = st.secrets["database"]["schema"]
            
            success, message = st.session_state.db_connection.connect(
                db_user, db_pass, db_host, db_port, db_name, schema
            )
            
            if success:
                st.session_state.schema = schema
                return True
            else:
                st.error(f"Database connection failed: {message}")
                return False
                
        except KeyError as e:
            st.error(f"Missing database configuration: {e}")
            return False
    
    return True

# Geocoding service initialization
def initialize_geocode_service():
    try:
        google_api_key = st.secrets["google"]["api_key"]
        if 'geocode_service' not in st.session_state:
            st.session_state.geocode_service = GeocodeService(google_api_key)
        return st.session_state.geocode_service
    except KeyError:
        st.warning("Google Maps API key not found. Location search features unavailable.")
        st.session_state.geocode_service = None
        return None

# Session state initialization
def initialize_session_state():
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = 'condo'
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = {}
        for agent_type in AGENT_CONFIGS.keys():
            st.session_state.chat_messages[agent_type] = []
    
    if 'agents' not in st.session_state:
        st.session_state.agents = initialize_agents()
    
    if 'geocode_service' not in st.session_state:
        st.session_state.geocode_service = initialize_geocode_service()
    
    if 'parser' not in st.session_state:
        st.session_state.parser = CrossAgentQueryParser()

# Process user query
async def process_user_query(query: str, agent_type: str) -> str:
    try:
        # Check for cross-agent query
        parsed = st.session_state.parser.parse_query(query)
        
        if parsed['type'] == 'single_agent':
            # Single agent query
            agent = st.session_state.agents.get(agent_type)
            if not agent:
                return f"Error: Agent {agent_type} not found"
            
            # Build conversation context
            conversation_context = ""
            chat_history = st.session_state.chat_messages.get(agent_type, [])
            if len(chat_history) > 1:
                recent_messages = chat_history[-4:]
                context_parts = []
                
                for msg in recent_messages:
                    if msg['role'] == 'user':
                        context_parts.append(f"User: {msg['content']}")
                    elif msg['role'] == 'assistant' and len(msg['content']) < 200:
                        context_parts.append(f"Assistant: {msg['content']}")
                
                if context_parts:
                    conversation_context = "\n".join(context_parts)
            
            # Enhanced query with context
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""CONVERSATION CONTEXT:
{conversation_context}

CURRENT REQUEST: {query}

Use context appropriately for follow-up questions."""
            
            # Clear any previous visualization
            if 'last_visualization' in st.session_state:
                del st.session_state.last_visualization
            
            # Use streaming with proper async handling
            result = Runner.run_streamed(agent, input=enhanced_query)
            
            # Stream the response with proper token handling
            full_response = ""
            response_container = st.empty()
            
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    full_response += event.data.delta
                    response_container.markdown(full_response + "▌")
            
            # Final response without cursor
            response_container.markdown(full_response)
            
            return full_response
            
        elif parsed['type'] == 'comparison':
            # Cross-agent comparison
            return await process_cross_agent_comparison(parsed, query)
        
        else:
            return f"Error: {parsed.get('error', 'Unknown cross-agent query type')}"
            
    except Exception as e:
        return f"Error processing query: {str(e)}"

async def process_cross_agent_comparison(parsed: dict, original_query: str) -> str:
    """Process cross-agent comparison queries"""
    try:
        agents = parsed['agents']
        if len(agents) < 2:
            return "Error: Need at least 2 agents for comparison"
        
        # Remove cross-agent syntax from query
        clean_query = re.sub(r'^#\w+(\s+vs\s+\w+)+\s*', '', original_query, flags=re.IGNORECASE)
        
        results = {}
        errors = []
        
        for agent_type in agents:
            try:
                agent = st.session_state.agents.get(agent_type)
                if not agent:
                    errors.append(f"Agent {agent_type} not found")
                    continue
                
                # Generate agent-specific query
                agent_query = f"Analyze {agent_type} properties for: {clean_query}"
                
                # Run agent
                result = Runner.run_streamed(agent, input=agent_query)
                
                # Collect response (non-streaming for comparison)
                response_text = ""
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        response_text += event.data.delta
                
                results[agent_type] = response_text
                
            except Exception as e:
                errors.append(f"{agent_type}: {str(e)}")
        
        # Format combined response
        if not results:
            return f"All agents failed. Errors: {'; '.join(errors)}"
        
        # Create comparison summary
        summary_parts = []
        summary_parts.append(f"🔗 **Cross-Agent Comparison Analysis**")
        summary_parts.append(f"📋 **Query:** {clean_query}")
        summary_parts.append(f"🤖 **Agents:** {', '.join([a.title() for a in results.keys()])}")
        summary_parts.append("")
        
        # Add results from each agent
        for agent_type, response in results.items():
            config = AGENT_CONFIGS[agent_type]
            summary_parts.append(f"### {config['icon']} {config['name']} Analysis")
            
            # Truncate long responses
            if len(response) > 500:
                response = response[:500] + "..."
            
            summary_parts.append(response)
            summary_parts.append("")
        
        # Add errors if any
        if errors:
            summary_parts.append("### ⚠️ **Warnings**")
            for error in errors:
                summary_parts.append(f"- {error}")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Error in cross-agent comparison: {str(e)}"

# UI Components
def render_agent_selection():
    """Render agent selection interface"""
    st.markdown('<div class="section-header">🤖 Select Property Expert</div>', unsafe_allow_html=True)
    
    # Agent selection cards
    cols = st.columns(3)
    
    for i, (agent_type, config) in enumerate(AGENT_CONFIGS.items()):
        with cols[i % 3]:
            # Get chat status
            message_count = len(st.session_state.chat_messages.get(agent_type, []))
            status_indicator = "💬" if message_count > 0 else "💤"
            
            # Create the card with button
            if st.button(
                f"{config['icon']} {config['name']}\n{config['description']}\n{status_indicator} {message_count} messages", 
                key=f"agent_{agent_type}",
                use_container_width=True
            ):
                st.session_state.current_agent = agent_type
                st.rerun()
    
    # Show selected agent
    if st.session_state.current_agent:
        current_config = AGENT_CONFIGS[st.session_state.current_agent]
        st.markdown(
            f'<div class="success-box">✅ Active Agent: {current_config["icon"]} {current_config["name"]}</div>', 
            unsafe_allow_html=True
        )
        
        # Show cross-agent syntax help
        with st.expander("🔗 Cross-Agent Query Syntax", expanded=False):
            st.markdown("""
            **Cross-Agent Query Examples:**
            
            **Comparison:**
            - `#condo vs office` - Compare condo and office properties
            - `#hotel vs retail vs office` - Compare three property types
            
            **Consultation:**
            - `#condo consult hospital` - Condo agent consults hospital data
            - `#office consult retail hospital` - Office agent consults multiple agents
            
            **Market Analysis:**
            - `#all market analysis Jakarta` - All agents analyze Jakarta market
            - `#all investment opportunities` - Comprehensive analysis
            
            **Impact Analysis:**
            - `#hospital impact condo` - How hospitals affect condo values
            - `#retail impact office` - Retail impact on office spaces
            """)

def render_ai_chat():
    """Render AI chat interface"""
    st.markdown('<div class="section-header">💬 AI Chat</div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    # Initialize services
    geocode_service = initialize_geocode_service()
    
    # Agent status display
    current_config = AGENT_CONFIGS[st.session_state.current_agent]
    st.markdown(f"""
    <div class="agent-status">
        {current_config['icon']} {current_config['name']}
    </div>
    """, unsafe_allow_html=True)
    
    # Get current agent's chat history
    current_history = st.session_state.chat_messages.get(st.session_state.current_agent, [])
    
    # Display chat history with visualizations
    for i, message in enumerate(current_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Redisplay visualization if present
            if message.get("visualization"):
                viz = message["visualization"]
                if viz["type"] in ["map", "nearby_map"]:
                    st.plotly_chart(viz["figure"], use_container_width=True)
                    
                    # Show additional info for nearby maps
                    if viz["type"] == "nearby_map":
                        st.caption(f"📍 {viz['location']} • Radius: {viz['radius']} km • Found: {viz['count']} properties")
                        
                elif viz["type"] == "chart":
                    st.plotly_chart(viz["figure"], use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me about properties..."):
        # Add user message
        current_history.append({
            "role": "user", 
            "content": prompt
        })
        st.session_state.chat_messages[st.session_state.current_agent] = current_history
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            # Show cross-agent indicator if detected
            if prompt.startswith('#'):
                st.markdown('<div class="cross-agent-indicator">🔗 Cross-Agent Query Detected</div>', unsafe_allow_html=True)
            
            # Process the query asynchronously
            with st.spinner("🤖 Processing ..."):
                try:
                    # Simple asyncio handling for Streamlit
                    try:
                        # Try to use existing event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is already running, create a new one
                            import nest_asyncio
                            nest_asyncio.apply()
                            response = asyncio.run(
                                process_user_query(prompt, st.session_state.current_agent)
                            )
                        else:
                            response = loop.run_until_complete(
                                process_user_query(prompt, st.session_state.current_agent)
                            )
                    except RuntimeError:
                        # No event loop exists, create one
                        response = asyncio.run(
                            process_user_query(prompt, st.session_state.current_agent)
                        )
                    
                    # Store visualization for redisplay
                    viz_data = st.session_state.get('last_visualization', None)
                    
                    # Add assistant response to history
                    current_history.append({
                        "role": "assistant",
                        "content": response,
                        "visualization": viz_data
                    })
                    st.session_state.chat_messages[st.session_state.current_agent] = current_history
                    
                    # If there's a visualization, rerun to display it
                    if viz_data:
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    response = f"Error: {str(e)}"
    
    # Chat management
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages[st.session_state.current_agent] = []
            st.rerun()
    
    with col2:
        if st.button("Show Last Data", use_container_width=True):
            if hasattr(st.session_state, 'last_query_result') and st.session_state.last_query_result is not None:
                with st.expander("Last Query Results", expanded=True):
                    st.dataframe(st.session_state.last_query_result, use_container_width=True)
            else:
                st.info("No previous query results available")
    
    with col3:
        if st.button("Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "agent": st.session_state.current_agent,
                "chat_history": st.session_state.chat_messages[st.session_state.current_agent]
            }
            
            st.download_button(
                label="Download Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"chat_{st.session_state.current_agent}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🏢 RHR Multi-Agent Property AI</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Check authentication
    if not check_authentication():
        login()
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to:", [
        "🤖 Agent Selection", 
        "💬 AI Chat",
        # "📝 Examples"
    ])
    
    # Show current user
    st.sidebar.markdown("---")
    st.sidebar.success(f"👤 Logged in as: {st.secrets['auth']['username']}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # Render selected page
    if page == "🤖 Agent Selection":
        render_agent_selection()
    elif page == "💬 AI Chat":
        render_ai_chat()
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 System Status**")
    
    # Database status
    if hasattr(st.session_state, 'db_connection') and st.session_state.db_connection.connection_status:
        st.sidebar.success("✅ Database Connected")
    else:
        st.sidebar.error("❌ Database Disconnected")
    
    # Agent status
    if hasattr(st.session_state, 'agents') and st.session_state.agents:
        st.sidebar.success("✅ Agents Ready")
    else:
        st.sidebar.error("❌ Agents Not Ready")
    
    # Current agent
    if st.session_state.current_agent:
        config = AGENT_CONFIGS[st.session_state.current_agent]
        st.sidebar.success(f"🤖 Agent: {config['icon']} {config['name']}")
    
    # Chat status
    total_messages = sum(len(msgs) for msgs in st.session_state.chat_messages.values())
    active_chats = sum(1 for msgs in st.session_state.chat_messages.values() if len(msgs) > 0)
    st.sidebar.info(f"💬 Chats: {active_chats} active, {total_messages} total messages")
    
    # Agent status breakdown
    st.sidebar.markdown("**🤖 Agent Status**")
    for agent_type, config in AGENT_CONFIGS.items():
        count = len(st.session_state.chat_messages.get(agent_type, []))
        status = "💬" if count > 0 else "💤"
        st.sidebar.text(f"{status} {config['icon']} {config['name']}: {count}")

if __name__ == "__main__":
    main()