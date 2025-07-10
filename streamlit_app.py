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

# Agent configurations
AGENT_CONFIGS = {
    'condo': {
        'name': 'Condo Expert',
        'icon': '🏠',
        'table': 'condo_converted_2025',
        'color': '#3498db',
        'description': 'Residential condominium specialist'
    },
    'hotel': {
        'name': 'Hotel Expert', 
        'icon': '🏨',
        'table': 'hotel_converted_2025',
        'color': '#e74c3c',
        'description': 'Hospitality property specialist'
    },
    'hospital': {
        'name': 'Hospital Expert',
        'icon': '🏥',
        'table': 'hospital_converted_2025',
        'color': '#9b59b6',
        'description': 'Healthcare facility specialist'
    },
    'office': {
        'name': 'Office Expert',
        'icon': '🏢',
        'table': 'office_converted_2025',
        'color': '#f39c12',
        'description': 'Commercial office specialist'
    },
    'retail': {
        'name': 'Retail Expert',
        'icon': '🏬',
        'table': 'retail_converted_2025',
        'color': '#27ae60',
        'description': 'Retail property specialist'
    },
    'land': {
        'name': 'Land Market Expert',
        'icon': '🌍',
        'table': 'engineered_property_data',
        'color': '#8b4513',
        'description': 'Land market and property value specialist'
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

# Function tools for agents
@function_tool
def execute_sql_query(sql_query: str) -> str:
    """Execute SQL query and return formatted results"""
    try:
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            
            # Display results in expandable section
            with st.expander("📊 Query Results", expanded=False):
                st.code(sql_query, language="sql")
                st.dataframe(result_df, use_container_width=True)
            
            # Return formatted summary
            if len(result_df) == 1 and len(result_df.columns) == 1:
                value = result_df.iloc[0, 0]
                return f"Query result: {value}"
            elif len(result_df) <= 10:
                return f"Query returned {len(result_df)} rows:\n{result_df.to_string(index=False)}"
            else:
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
        
        # Handle different coordinate formats based on agent type
        if st.session_state.current_agent == 'land':
            map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
            map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
        else:
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
        
        # Create map
        fig = go.Figure()
        
        # Create hover text based on agent type
        hover_text = []
        for idx, row in map_df.iterrows():
            text_parts = []
            if 'id' in row and pd.notna(row['id']):
                text_parts.append(f"ID: {row['id']}")
            
            # Add relevant fields based on available columns
            for col in ['project_name', 'building_name', 'object_name', 'alamat']:
                if col in row and pd.notna(row[col]):
                    text_parts.append(f"Name: {row[col]}")
                    break
            
            if 'grade' in row and pd.notna(row['grade']):
                text_parts.append(f"Grade: {row['grade']}")
            
            if 'wadmpr' in row and pd.notna(row['wadmpr']):
                text_parts.append(f"Province: {row['wadmpr']}")
            
            hover_text.append("<br>".join(text_parts))
        
        # Use agent-specific color
        agent_color = AGENT_CONFIGS[st.session_state.current_agent]['color']
        
        # Add markers
        fig.add_trace(go.Scattermapbox(
            lat=map_df['latitude'],
            lon=map_df['longitude'],
            mode='markers',
            marker=dict(size=8, color=agent_color),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name=f'{st.session_state.current_agent.title()} Properties'
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
        
        # Store for redisplay
        st.session_state.last_visualization = {
            "type": "map",
            "figure": fig,
            "title": title
        }

        # Display map
        st.plotly_chart(fig, use_container_width=True)
        
        # Store for future reference
        st.session_state.last_query_result = map_df.copy()
        
        # Show query details
        with st.expander("🗺️ Map Query Details", expanded=False):
            st.code(sql_query, language="sql")
            st.info(f"Mapped {len(map_df)} properties with valid coordinates")
        
        return f"✅ Map successfully created with {len(map_df)} {st.session_state.current_agent} properties"
        
    except Exception as e:
        return f"Error creating map: {str(e)}"

@function_tool
def create_chart_visualization(chart_type: str, sql_query: str, title: str, 
                              x_column: str = None, y_column: str = None, 
                              color_column: str = None) -> str:
    """Create chart visualizations from SQL query results"""
    try:
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is None or len(result_df) == 0:
            return f"Error: No data returned from query - {query_msg}"
        
        # Auto-detect columns if not provided
        if x_column is None or x_column not in result_df.columns:
            x_column = result_df.columns[0]
        if y_column is None or y_column not in result_df.columns:
            numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
            y_column = numeric_cols[0] if numeric_cols else result_df.columns[1] if len(result_df.columns) > 1 else None
        
        fig = None
        
        # Create chart based on type
        if chart_type == "bar":
            fig = px.bar(result_df, x=x_column, y=y_column, color=color_column, title=title)
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == "pie":
            if y_column:
                fig = px.pie(result_df, names=x_column, values=y_column, title=title)
            else:
                pie_data = result_df[x_column].value_counts().reset_index()
                pie_data.columns = [x_column, 'count']
                fig = px.pie(pie_data, names=x_column, values='count', title=title)
        elif chart_type == "line":
            fig = px.line(result_df, x=x_column, y=y_column, color=color_column, title=title, markers=True)
        elif chart_type == "scatter":
            fig = px.scatter(result_df, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(result_df, x=x_column if x_column else y_column, color=color_column, title=title)
        else:
            fig = px.bar(result_df, x=x_column, y=y_column, color=color_column, title=title)
            fig.update_layout(xaxis_tickangle=-45)
        
        if fig:
            fig.update_layout(
                height=500,
                template="plotly_white",
                title_x=0.5,
                margin=dict(l=50, r=50, t=80, b=100)
            )
            
            # Store for redisplay
            st.session_state.last_visualization = {
                "type": "chart",
                "figure": fig,
                "chart_type": chart_type,
                "title": title
            }
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            
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
            title = f"{st.session_state.current_agent.title()} properties within {radius_km} km from {location_name}"
        
        # Geocode the location
        lat, lng, formatted_address = st.session_state.geocode_service.geocode_address(location_name)
        
        if lat is None or lng is None:
            return f"Error: Could not find coordinates for location '{location_name}'. Try being more specific."
        
        st.success(f"📍 Location found: {formatted_address}")
        st.info(f"Coordinates: {lat:.6f}, {lng:.6f}")
        
        # Get current agent's table
        table_name = AGENT_CONFIGS[st.session_state.current_agent]['table']
        
        # Determine coordinate columns based on agent type
        if st.session_state.current_agent == 'land':
            lat_col = "CAST(latitude AS NUMERIC)"
            lng_col = "CAST(longitude AS NUMERIC)" 
            coord_filter = "latitude IS NOT NULL AND longitude IS NOT NULL AND latitude != '' AND longitude != ''"
        else:
            lat_col = "latitude"
            lng_col = "longitude"
            coord_filter = "latitude IS NOT NULL AND longitude IS NOT NULL AND latitude != 0 AND longitude != 0"
        
        # Query nearby properties using Haversine formula
        sql_query = f"""
        SELECT 
            id,
            {lat_col} as latitude,
            {lng_col} as longitude,
            *,
            (6371 * acos(
                cos(radians({lat})) * cos(radians({lat_col})) * 
                cos(radians({lng_col}) - radians({lng})) + 
                sin(radians({lat})) * sin(radians({lat_col}))
            )) as distance_km
        FROM {table_name}
        WHERE 
            {coord_filter}
            AND (6371 * acos(
                cos(radians({lat})) * cos(radians({lat_col})) * 
                cos(radians({lng_col}) - radians({lng})) + 
                sin(radians({lat})) * sin(radians({lat_col}))
            )) <= {radius_km}
        ORDER BY distance_km ASC
        LIMIT 50
        """
        
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
            
            # Add property markers
            hover_text = []
            for idx, row in result_df.iterrows():
                text_parts = [f"ID: {row['id']}"]
                
                # Add name field if available
                for col in ['project_name', 'building_name', 'object_name', 'alamat']:
                    if col in row and pd.notna(row[col]):
                        text_parts.append(f"Name: {row[col]}")
                        break
                
                if 'wadmpr' in row and pd.notna(row['wadmpr']):
                    text_parts.append(f"Province: {row['wadmpr']}")
                
                text_parts.append(f"Distance: {row['distance_km']:.2f} km")
                hover_text.append("<br>".join(text_parts))
            
            # Use agent-specific color
            agent_color = AGENT_CONFIGS[st.session_state.current_agent]['color']
            
            fig.add_trace(go.Scattermapbox(
                lat=result_df['latitude'],
                lon=result_df['longitude'],
                mode='markers',
                marker=dict(size=8, color=agent_color),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=f'Properties ({len(result_df)})'
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
            
            # Store for redisplay
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
            
            # Show results table
            with st.expander("📊 Nearby Properties Details", expanded=False):
                st.code(sql_query, language="sql")
                display_cols = ['id'] + [col for col in result_df.columns 
                                    if col not in ['latitude', 'longitude', 'geometry']]
                st.dataframe(result_df[display_cols].round(2), use_container_width=True)

            return f"✅ Found {len(result_df)} {st.session_state.current_agent} properties within {radius_km} km from {location_name}. Closest property is {result_df['distance_km'].min():.2f} km away."
        
        else:
            return f"❌ No {st.session_state.current_agent} properties found within {radius_km} km from {location_name}."
        
    except Exception as e:
        return f"Error finding nearby properties: {str(e)}"

def initialize_agents():
    """Initialize all property agents using o4-mini"""
    
    # Set OpenAI API key
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
    except KeyError:
        st.error("OpenAI API key not found in secrets.toml")
        return None
    
    agents = {}
    
    # Agent-specific instructions with detailed column information
    AGENT_INSTRUCTIONS = {
        'condo': f"""You are a Condominium Property Expert AI for RHR specializing in residential condominiums using o4-mini.
    Table: {table_name}

    CONDO EXPERTISE:
    - Residential condominium analysis
    - Developer performance and delivery
    - Unit counts and residential capacity
    - Condo grades and market positioning

    COLUMN DETAILS:
    - id (INTEGER), geometry (TEXT), latitude/longitude (DOUBLE PRECISION)
    - project_name (TEXT), address (TEXT), developer (TEXT)
    - grade (TEXT), unit (INTEGER), project_status (TEXT)
    - wadmpr (TEXT), wadmkk (TEXT), wadmkc (TEXT)

    SQL RULES:
    - unit is INTEGER - use directly: SUM(unit), AVG(unit)
    - NO PRICING DATA available in this table
    - For maps: Include id, latitude, longitude, project_name, address, grade
    - For capacity: SUM(unit) as total_units, COUNT(*) as project_count
    - Always include LIMIT to prevent large results

    RESPONSE STYLE:
    - Always respond in user's language (auto-detect)
    - Provide residential market insights with data
    - Use tools appropriately based on request type
    - Handle follow-up questions using context

    CRITICAL: You can ONLY answer questions in this condo property domain scope!""",

        'hotel': f"""You are a Hotel Property Expert AI for RHR specializing in hospitality properties using o4-mini.
    Table: {table_name}

    HOTEL EXPERTISE:
    - Hotel and hospitality analysis
    - Star rating classifications (1-5 stars)
    - Hotel management and operations
    - Event facilities and capacity

    COLUMN DETAILS:
    - id (INTEGER), geometry (TEXT), latitude/longitude (DOUBLE PRECISION)
    - project_name (TEXT), address (TEXT), developer (TEXT), management (TEXT)
    - star (TEXT), concept (TEXT), unit_developed (INTEGER), ballroom_capacity (INTEGER)
    - price_2016 to price_2025 (TEXT), price_avg (TEXT)
    - wadmpr (TEXT), wadmkk (TEXT), wadmkc (TEXT)

    SQL RULES:
    - unit_developed, ballroom_capacity are INTEGER - use directly
    - Price columns are TEXT - use CAST(price_avg AS NUMERIC)
    - For maps: Include id, latitude, longitude, project_name, star, concept
    - For capacity: SUM(unit_developed), AVG(ballroom_capacity)
    - Always include LIMIT to prevent large results

    RESPONSE STYLE:
    - Always respond in user's language (auto-detect)
    - Provide hospitality insights with data
    - Use tools appropriately based on request type
    - Handle follow-up questions using context

    CRITICAL: You can ONLY answer questions in this hotel property domain scope!""",

        'office': f"""You are an Office Property Expert AI for RHR specializing in commercial office spaces using o4-mini.
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
    - sga/gfa are TEXT - use CAST(sga AS NUMERIC) if needed
    - Always include LIMIT to prevent large results

    RESPONSE STYLE:
    - Always respond in user's language (auto-detect)
    - Provide office market insights with data
    - Use tools appropriately based on request type
    - Handle follow-up questions using context

    CRITICAL: You can ONLY answer questions in this office property domain scope!""",

        'hospital': f"""You are a Hospital Property Expert AI for RHR specializing in healthcare facilities using o4-mini.
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
    - For capacity: SUM(beds_capacity) WHERE beds_capacity IS NOT NULL
    - Always include LIMIT to prevent large results

    RESPONSE STYLE:
    - Always respond in user's language (auto-detect)
    - Provide healthcare insights with data
    - Use tools appropriately based on request type
    - Handle follow-up questions using context

    CRITICAL: You can ONLY answer questions in this hospital property domain scope!""",

        'retail': f"""You are a Retail Property Expert AI for RHR specializing in commercial retail spaces using o4-mini.
    Table: {table_name}

    RETAIL EXPERTISE:
    - Shopping malls, retail outlets, commercial spaces
    - Net Lettable Area (NLA) and Gross Floor Area (GFA) analysis
    - Retail pricing trends and market analysis
    - Developer and project performance

    COLUMN DETAILS:
    - id (INTEGER), geometry (TEXT), latitude/longitude (DOUBLE PRECISION)
    - project_name (TEXT), address (TEXT), developer (TEXT)
    - grade (TEXT), nla (TEXT), gfa (TEXT)
    - price_2016 to price_2025 (TEXT), price_avg (TEXT)
    - wadmpr (TEXT), wadmkk (TEXT), wadmkc (TEXT)

    SQL RULES:
    - Price columns are TEXT - use CAST(price_avg AS NUMERIC) for calculations
    - nla and gfa are TEXT - use CAST(nla AS NUMERIC) if needed
    - For maps: Include id, latitude, longitude, project_name, address, grade
    - For price analysis: CAST(price_avg AS NUMERIC) WHERE price_avg IS NOT NULL AND price_avg != ''
    - Always include LIMIT to prevent large results

    RESPONSE STYLE:
    - Always respond in user's language (auto-detect)
    - Provide retail market insights with data
    - Use tools appropriately based on request type
    - Handle follow-up questions using context

    CRITICAL: You can ONLY answer questions in this retail property domain scope!""",

        'land': f"""You are a Land Market Expert AI for RHR specializing in land property analysis using o4-mini.
    Table: {table_name}

    LAND EXPERTISE:
    - Land valuation and price per meter analysis
    - Land characteristics and development potential
    - Geographic market trends and accessibility
    - Area-based pricing comparison

    COLUMN DETAILS:
    - id (INTEGER), alamat (TEXT), latitude/longitude (TEXT)
    - luas_tanah (FLOAT), hpm (FLOAT), tahun_pengambilan_data (INTEGER)
    - bentuk_tapak (TEXT), posisi_tapak (TEXT), orientasi (TEXT)
    - wadmpr (TEXT), wadmkk (TEXT), wadmkc (TEXT), wadmkd (TEXT)

    SQL RULES:
    - hpm, luas_tanah are FLOAT - use directly: AVG(hpm), SUM(luas_tanah)
    - latitude/longitude are TEXT - use CAST(latitude AS NUMERIC)
    - For maps: Include id, CAST(latitude AS NUMERIC), CAST(longitude AS NUMERIC), alamat, hpm
    - Price filtering: WHERE hpm IS NOT NULL AND hpm > 0
    - Always include LIMIT to prevent large results

    RESPONSE STYLE:
    - Always respond in user's language (auto-detect)
    - Provide land market insights with data
    - Use tools appropriately based on request type
    - Handle follow-up questions using context

    CRITICAL: You can ONLY answer questions in this land property domain scope!"""
    }

    # Create specialized agents for each property type
    for agent_type, config in AGENT_CONFIGS.items():
        table_name = config['table']
        
        # Get agent-specific instructions with table name formatting
        agent_instructions = AGENT_INSTRUCTIONS[agent_type].format(table_name=table_name)
        
        # Add common tool instructions
        full_instructions = f"""{agent_instructions}

    AVAILABLE TOOLS:
    1. execute_sql_query(sql_query) - Run SQL queries and display results
    2. create_map_visualization(sql_query, title) - Create location maps
    3. create_chart_visualization(chart_type, sql_query, title, x_column, y_column, color_column) - Create charts
    4. find_nearby_projects(location_name, radius_km, title) - Find projects near locations

    TOOL USAGE RULES:
    - If user asks for charts/graphs ("grafik", "chart", "barchart", "pie", etc.), use create_chart_visualization
    - If user asks for properties near a location, use find_nearby_projects
    - If user asks for a map, use create_map_visualization  
    - Otherwise use execute_sql_query for data analysis

    Generate appropriate SQL queries and use tools based on the user's request."""
        
        # Create agent
        agent = Agent(
            name=f"{agent_type}_expert",
            instructions=full_instructions,
            model="o4-mini",
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
        {current_config['icon']} {current_config['name']} - Powered by o4-mini
    </div>
    """, unsafe_allow_html=True)
    
    # Get current agent's chat history
    current_history = st.session_state.chat_messages.get(st.session_state.current_agent, [])
    
    # Display service status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.db_connection.connection_status:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Disconnected")
    
    with col2:
        if geocode_service:
            st.success("✅ Location Service Active")
        else:
            st.warning("⚠️ Location Service Inactive")
    
    with col3:
        if st.session_state.agents:
            st.success("✅ o4-mini Agents Ready")
        else:
            st.error("❌ Agents Not Ready")
    
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
            with st.spinner("🤖 Processing with o4-mini..."):
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
                "model": "o4-mini",
                "chat_history": st.session_state.chat_messages[st.session_state.current_agent]
            }
            
            st.download_button(
                label="Download Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"chat_{st.session_state.current_agent}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

def render_examples():
    """Render example queries to help users"""
    st.markdown('<div class="section-header">📝 Example Queries</div>', unsafe_allow_html=True)
    
    # Current agent examples
    current_agent = st.session_state.current_agent
    current_config = AGENT_CONFIGS[current_agent]
    
    st.markdown(f"### {current_config['icon']} {current_config['name']} Examples")
    
    agent_examples = {
        'condo': [
            "Berapa total unit condo di Jakarta?",
            "Siapa developer terbesar untuk proyek condo?",
            "Buatkan peta semua proyek condo di Bali",
            "Grafik bar developer vs jumlah unit",
            "Proyek condo terdekat dari Mall Taman Anggrek"
        ],
        'hotel': [
            "Berapa hotel bintang 5 di Indonesia?",
            "Buatkan peta hotel di Yogyakarta",
            "Grafik pie distribusi hotel per bintang",
            "Hotel terdekat dari Monas radius 2km",
            "Siapa management hotel terbesar?"
        ],
        'office': [
            "Berapa rata-rata harga office Grade A?",
            "Buatkan peta office building di Jakarta",
            "Grafik harga office per tahun",
            "Office terdekat dari Sudirman radius 1km",
            "Perbandingan harga office Grade A vs B"
        ],
        'hospital': [
            "Berapa total kapasitas tempat tidur rumah sakit?",
            "Buatkan peta rumah sakit di Surabaya",
            "Grafik distribusi rumah sakit per grade",
            "Rumah sakit terdekat dari Senayan",
            "Rumah sakit yang menerima BPJS"
        ],
        'retail': [
            "Berapa rata-rata harga retail per meter?",
            "Buatkan peta retail space di Bandung",
            "Grafik harga retail per tahun",
            "Retail terdekat dari Plaza Indonesia",
            "Developer retail terbesar di Indonesia"
        ],
        'land': [
            "Berapa harga tanah rata-rata per meter di Jakarta?",
            "Buatkan peta tanah di Bekasi",
            "Grafik harga tanah per provinsi",
            "Tanah terdekat dari Bogor radius 5km",
            "Perbandingan harga tanah per orientasi"
        ]
    }
    
    examples = agent_examples.get(current_agent, [])
    
    for i, example in enumerate(examples):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"• {example}")
        with col2:
            if st.button("Try", key=f"try_{current_agent}_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
    
    # Cross-agent examples
    st.markdown("### 🔗 Cross-Agent Examples")
    
    cross_agent_examples = [
        "#condo vs hotel - Compare condo and hotel properties",
        "#office vs retail - Compare office and retail spaces",
        "#hospital vs condo - Compare hospital and condo locations",
        "#land vs office - Compare land prices and office locations"
    ]
    
    for i, example in enumerate(cross_agent_examples):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"• {example}")
        with col2:
            if st.button("Try", key=f"try_cross_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

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
        "📝 Examples"
    ])
    
    # Show current user
    st.sidebar.markdown("---")
    st.sidebar.success(f"👤 Logged in as: {st.secrets['auth']['username']}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Handle example query injection
    if hasattr(st.session_state, 'example_query'):
        # Switch to chat tab and inject query
        page = "💬 AI Chat"
        
        # Add to chat messages
        current_history = st.session_state.chat_messages.get(st.session_state.current_agent, [])
        current_history.append({
            "role": "user", 
            "content": st.session_state.example_query
        })
        st.session_state.chat_messages[st.session_state.current_agent] = current_history
        
        # Clear the example query
        del st.session_state.example_query
        st.rerun()
    
    # Render selected page
    if page == "🤖 Agent Selection":
        render_agent_selection()
    elif page == "💬 AI Chat":
        render_ai_chat()
    elif page == "📝 Examples":
        render_examples()
    
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
        st.sidebar.success("✅ o4-mini Agents Ready")
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