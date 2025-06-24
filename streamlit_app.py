import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
import openai
import json
import traceback
from typing import Optional, Dict, List, Tuple, Any
import warnings
import asyncio
import aiohttp
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import time
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import folium
from streamlit_folium import st_folium

# import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RHR Market Research Agent",
    page_icon="🏠",
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
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #efe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .market-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .data-table-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .data-table-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .selected-table {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class DatabaseConnection:
    """Handle PostgreSQL/PostGIS database connections"""
    
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
            st.session_state.schema = schema
            return True, "Connection successful!"
        
        except Exception as e:
            self.connection_status = False
            return False, f"Connection failed: {str(e)}"
    
    def execute_query(self, query: str) -> tuple:
        """Execute SQL query and return results"""
        try:
            if not self.connection_status:
                return None, "No database connection established"
            
            df = pd.read_sql(query, self.engine)
            return df, "Query executed successfully"
        
        except Exception as e:
            return None, f"Query execution failed: {str(e)}"
    
    def get_table_columns(self, table_name: str, schema: str = 'public') -> tuple:
        """Get column information for a specific table"""
        try:
            query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' AND table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            df, msg = self.execute_query(query)
            return df, msg
        except Exception as e:
            return None, f"Failed to get table columns: {str(e)}"
    
    def get_column_unique_values(self, table_name: str, column_name: str, schema: str = 'public', limit: int = 1000) -> tuple:
        """Get unique values for a specific column"""
        try:
            query = f"""
            SELECT DISTINCT "{column_name}" as unique_value
            FROM "{schema}"."{table_name}"
            WHERE "{column_name}" IS NOT NULL
            ORDER BY "{column_name}"
            LIMIT {limit}
            """
            df, msg = self.execute_query(query)
            if df is not None:
                return df['unique_value'].tolist(), msg
            return None, msg
        except Exception as e:
            return None, f"Failed to get unique values: {str(e)}"

class DataChatbot:
    """AI chatbot for data analysis and exploration"""
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=api_key,
            temperature=0.3,
            max_tokens=2000,
            streaming=True
        )
    
    def generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate comprehensive data summary including more details"""
        
        # Basic info
        summary = f"COMPLETE DATASET ANALYSIS:\n"
        summary += f"- Total records: {len(df):,}\n"
        summary += f"- Total columns: {len(df.columns)}\n"
        summary += f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
        
        # ALL Column information
        summary += f"ALL COLUMNS: {list(df.columns)}\n\n"
        
        # Complete statistical analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "COMPLETE STATISTICAL ANALYSIS:\n"
            desc_stats = df[numeric_cols].describe()
            summary += desc_stats.to_string()
            summary += "\n\n"
        
        # Complete categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary += "COMPLETE CATEGORICAL ANALYSIS:\n"
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                summary += f"\n{col} (Total unique: {df[col].nunique()}):\n"
                summary += value_counts.to_string()
                summary += "\n"
        
        # Data quality for ALL records
        summary += f"\nDATA QUALITY ACROSS ALL {len(df):,} RECORDS:\n"
        missing_data = df.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                pct = (missing_count / len(df)) * 100
                summary += f"- {col}: {missing_count:,} missing ({pct:.1f}%)\n"
        
        return summary
    
    def create_system_prompt(self, df: pd.DataFrame) -> str:
        """Create comprehensive system prompt with FULL dataset"""
        
        data_summary = self.generate_data_summary(df)
        
        # Convert full dataset to JSON for AI analysis
        full_data_json = df.to_json(orient='records', date_format='iso')
        
        system_prompt = f"""
        You are an Indonesian expert data analyst assistant specializing in real estate and property data analysis. 
        You have access to the COMPLETE dataset with ALL {len(df):,} records for comprehensive analysis.
        
        DATASET SUMMARY:
        {data_summary}
        
        COMPLETE DATASET (JSON format):
        {full_data_json}
        
        YOU CAN PERFORM:
        1. Analysis on ALL {len(df):,} records, not just samples
        2. Calculations across the entire dataset
        3. Specific property lookups by any criteria
        4. Grouping and aggregation analysis
        5. Statistical analysis on complete data
        6. Pattern identification across all records
        7. Outlier detection in the full dataset
        8. Correlation analysis on all data points
        
        ANALYSIS CAPABILITIES:
        - Calculate statistics on any subset of the data
        - Find specific properties matching criteria
        - Identify highest/lowest values across all records
        - Perform grouping analysis (by location, price range, etc.)
        - Detect patterns and anomalies in the complete dataset
        - Compare properties across the entire dataset
        - Generate insights based on ALL available data
        
        INSTRUCTIONS:
        - Always analyze the COMPLETE dataset, not just samples
        - Provide specific numbers and examples from the full data
        - When asked about "all properties" or "total", use the complete dataset
        - Reference specific records when providing examples
        - Perform calculations across all {len(df):,} records
        - Base all insights on the complete dataset analysis
        - You can only answer with Bahasa Indonesia
        
        Remember: You have access to every single record in this dataset for comprehensive analysis.
        """
        
        return system_prompt

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = DatabaseConnection()
    
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None
    
    if 'table_columns' not in st.session_state:
        st.session_state.table_columns = None
    
    if 'applied_filters' not in st.session_state:
        st.session_state.applied_filters = {}

def get_api_key(key_name: str = "main") -> str:
    """Get OpenAI API key from various sources"""
    # Try to get from secrets first
    try:
        secrets_api_key = st.secrets["openai"]["api_key"]
        if secrets_api_key:
            return secrets_api_key
    except:
        pass
    
    # Fallback to environment variable
    env_api_key = os.getenv('OPENAI_API_KEY')
    if env_api_key:
        st.success("✅ Using OpenAI API key from environment variable")
        return env_api_key
    
    # Last resort - manual input
    st.info("💡 You can set your API key in `.streamlit/secrets.toml` or environment variable for security")
    return st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key=f"api_key_{key_name}")

def render_database_connection():
    """Render database connection section"""
    st.markdown('<div class="section-header">🔗 Koneksi Database</div>', unsafe_allow_html=True)
    
    # Get connection parameters from secrets and auto-connect
    try:
        db_user = st.secrets["database"]["user"]
        db_pass = st.secrets["database"]["password"]
        db_host = st.secrets["database"]["host"]
        db_port = st.secrets["database"]["port"]
        db_name = st.secrets["database"]["name"]
        schema = st.secrets["database"]["schema"]
        
        # st.success("✅ Database configuration loaded from secrets")
        
        # Auto-connect if not already connected
        if not st.session_state.db_connection.connection_status:
            with st.spinner("Connecting to database..."):
                success, message = st.session_state.db_connection.connect(
                    db_user, db_pass, db_host, db_port, db_name, schema
                )
                
                if success:
                    st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
                    
    except KeyError as e:
        st.error(f"❌ Missing database configuration in secrets.toml: {e}")
        st.info("Please ensure your `.streamlit/secrets.toml` file contains the [database] section with all required fields")
    except Exception as e:
        st.error(f"❌ Error loading database configuration: {e}")
    
    if st.session_state.db_connection.connection_status:
        st.markdown('<div class="success-box">🟢 Tersambung ke Database</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">🔴 Belum Ada Koneksi ke Database</div>', unsafe_allow_html=True)

def build_property_query(lat, lon, luas_tanah_range=None, lebar_jalan_range=None, kondisi_wilayah_opt=None, schema='public', table='engineered_property_data'):
    """
    Build SQL query for property search based on coordinates and filters
    """
    base_query = f"""
    SELECT *,
           ST_Distance(
               ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography,
               geometry::geography
           ) as distance_meters
    FROM "{schema}"."{table}"
    WHERE 1=1
    """
    
    conditions = []
    
    # Add luas_tanah filter
    if luas_tanah_range:
        min_luas, max_luas = luas_tanah_range
        conditions.append(f'"luas_tanah" BETWEEN {min_luas} AND {max_luas}')
    
    # Add lebar_jalan filter  
    if lebar_jalan_range:
        min_lebar, max_lebar = lebar_jalan_range
        conditions.append(f'"lebar_jalan_di_depan" BETWEEN {min_lebar} AND {max_lebar}')
    
    # Add kondisi_wilayah filter
    if kondisi_wilayah_opt and len(kondisi_wilayah_opt) > 0:
        kondisi_values = "', '".join(kondisi_wilayah_opt)
        conditions.append(f'"kondisi_wilayah_sekitar" IN (\'{kondisi_values}\')')
    
    # Add conditions to query
    if conditions:
        base_query += " AND " + " AND ".join(conditions)
    
    # Order by distance and limit to 300
    base_query += " ORDER BY distance_meters LIMIT 300"
    
    return base_query

def render_land_market_filtering():
    """Enhanced filtering for Land Market with two options"""
    
    if st.session_state.selected_table != 'engineered_property_data':
        return
    
    st.markdown("### 🔍 **Data Selection Method**")
    
    # Filter method selection
    filter_method = st.radio(
        "Choose your filtering method:",
        ["📍 Administrative Area Filtering", "🗺️ Point-Based Location Search"],
        help="Select how you want to filter the land market data"
    )
    
    db = st.session_state.db_connection
    schema = st.session_state.get('schema', 'public')
    table = st.session_state.selected_table
    
    if filter_method == "📍 Administrative Area Filtering":
        # Original administrative filtering (wadmkc method)
        render_administrative_filtering(db, schema, table)
    
    else:
        # New point-based filtering
        render_point_based_filtering(db, schema, table)

def render_administrative_filtering(db, schema, table):
    """Original administrative area filtering"""
    st.markdown("#### 🎯 **Administrative Area Filtering**")
    
    # Province (Required)
    st.markdown("**Required: Select Province/Region**")
    with st.spinner("Loading province options..."):
        province_values, province_msg = db.get_column_unique_values(table, 'wadmpr', schema)

    if province_values:
        selected_province = st.selectbox(
            "Choose a Province/Region:",
            [""] + province_values,
            help="You must select a province/region to continue",
            key="admin_province"
        )
        if not selected_province:
            st.error("❌ Please select a province/region to continue")
            return
    else:
        st.error(f"Could not load provinces: {province_msg}")
        return

    # Regency/City (Required)
    st.markdown("**Required: Select Regency/City**")
    with st.spinner("Loading regency/city options..."):
        regency_query = f'''
            SELECT DISTINCT wadmkk FROM "{schema}"."{table}"
            WHERE wadmpr = '{selected_province}'
            ORDER BY wadmkk
        '''
        regency_df, _ = db.execute_query(regency_query)
        regency_values = regency_df['wadmkk'].dropna().tolist() if regency_df is not None else []

    if regency_values:
        selected_regency = st.selectbox(
            "Choose a Regency/City:",
            [""] + regency_values,
            help="You must select a regency/city to continue",
            key="admin_regency"
        )
        if not selected_regency:
            st.error("❌ Please select a regency/city to continue")
            return
    else:
        st.error("No regency/city found for the selected province.")
        return

    # District (Required)
    st.markdown("**Required: Select District**")
    with st.spinner("Loading district options..."):
        district_query = f'''
            SELECT DISTINCT wadmkc FROM "{schema}"."{table}"
            WHERE wadmpr = '{selected_province}' AND wadmkk = '{selected_regency}'
            ORDER BY wadmkc
        '''
        district_df, _ = db.execute_query(district_query)
        district_values = district_df['wadmkc'].dropna().tolist() if district_df is not None else []

    if district_values:
        selected_district = st.selectbox(
            "Choose a District:",
            [""] + district_values,
            help="You must select a district to continue",
            key="admin_district"
        )
        if not selected_district:
            st.error("❌ Please select a district to continue")
            return
    else:
        st.error("No district found for the selected regency/city.")
        return

    # Subdistrict (Optional)
    st.markdown("**Optional: Select Subdistrict**")
    with st.spinner("Loading subdistrict options..."):
        subdistrict_query = f'''
            SELECT DISTINCT wadmkd FROM "{schema}"."{table}"
            WHERE wadmpr = '{selected_province}' AND wadmkk = '{selected_regency}' AND wadmkc = '{selected_district}'
            ORDER BY wadmkd
        '''
        subdistrict_df, _ = db.execute_query(subdistrict_query)
        subdistrict_values = subdistrict_df['wadmkd'].dropna().tolist() if subdistrict_df is not None else []

    selected_subdistrict = st.selectbox(
        "Choose a Subdistrict (optional):",
        [""] + subdistrict_values,
        key="admin_subdistrict"
    )

    # Build filters
    filters = {}
    filters['wadmpr'] = [selected_province]
    if selected_regency:
        filters['wadmkk'] = [selected_regency]
    if selected_district:
        filters['wadmkc'] = [selected_district]
    if selected_subdistrict:
        filters['wadmkd'] = [selected_subdistrict]

    # Additional filters (Year and HPM)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Filter by Year:**")
        area_where_parts = []
        for key in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']:
            if filters.get(key):
                area_where_parts.append(f'"{key}" = \'{filters[key][0]}\'')
        area_where_clause = " AND ".join(area_where_parts)
        
        years_query = f"""
            SELECT DISTINCT tahun_pengambilan_data 
            FROM "{schema}"."{table}"
            WHERE {area_where_clause}
            ORDER BY tahun_pengambilan_data
        """
        years_df, _ = db.execute_query(years_query)
        year_values = years_df['tahun_pengambilan_data'].dropna().tolist() if years_df is not None else []

        if year_values:
            selected_years = st.multiselect(
                "Select years:",
                year_values,
                default=year_values,
                help="Choose which years to include in analysis",
                key="admin_years"
            )
            if len(selected_years) < len(year_values):
                filters['tahun_pengambilan_data'] = selected_years

    with col2:
        st.markdown("**Filter by HPM (Price Range):**")
        hpm_query = f"""
            SELECT MIN("hpm") as min_hpm, MAX("hpm") as max_hpm 
            FROM "{schema}"."{table}"
            WHERE {area_where_clause} AND "hpm" IS NOT NULL
        """
        hpm_result, _ = db.execute_query(hpm_query)
        if hpm_result is not None and len(hpm_result) > 0:
            min_hpm = float(hpm_result['min_hpm'].iloc[0])
            max_hpm = float(hpm_result['max_hpm'].iloc[0])

            st.write(f"HPM range: {min_hpm:,.0f} - {max_hpm:,.0f}")

            hpm_range = st.slider(
                "Select HPM range:",
                min_value=min_hpm,
                max_value=max_hpm,
                value=(min_hpm, max_hpm),
                step=1000.0,
                help="Filter properties by price per square meter",
                key="admin_hpm"
            )
            if hpm_range != (min_hpm, max_hpm):
                filters['hpm'] = {'min': hpm_range[0], 'max': hpm_range[1], 'type': 'range'}

    # Load data button for administrative filtering
    if st.button("🎯 Get Data", type="primary", use_container_width=True, key="admin_get_data"):
        load_administrative_data(db, schema, table, filters)

def render_point_based_filtering(db, schema, table):
    """New point-based location filtering"""
    st.markdown("#### 🗺️ **Point-Based Location Search**")
    
    # Map for coordinate selection
    st.markdown("**Step 1: Select Location on Map**")
    
    # Create Indonesia-centered map
    indonesia_center = [-2.5, 118.0]  # Center of Indonesia
    m = folium.Map(
        location=indonesia_center,
        zoom_start=5,
        tiles="OpenStreetMap"
    )
    
    # Add instruction marker
    folium.Marker(
        indonesia_center,
        popup="Click anywhere on the map to select a location",
        tooltip="Indonesia Center - Click to select location",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    # Display map and get click data
    map_data = st_folium(m, width=700, height=400, key="location_map")
    
    # Handle map clicks
    selected_lat = None
    selected_lon = None

    if map_data['last_clicked']:
        selected_lat = map_data['last_clicked']['lat']
        selected_lon = map_data['last_clicked']['lng']
        
        # Add marker at clicked location
        folium.Marker(
            [selected_lat, selected_lon],
            popup=f"Selected: {selected_lat:.6f}, {selected_lon:.6f}",
            tooltip="Search Location",
            icon=folium.Icon(color='red', icon='map-pin')
        ).add_to(m)
        
        st.success(f"📍 Selected coordinates: {selected_lat:.6f}, {selected_lon:.6f}")
    
    # Manual coordinate input option
    st.markdown("**Alternative: Manual Coordinate Input**")
    col1, col2 = st.columns(2)
    
    with col1:
        manual_lat = st.number_input(
            "Latitude:",
            min_value=-90.0,
            max_value=90.0,
            value=selected_lat if selected_lat else -6.2088,
            format="%.6f",
            key="manual_lat"
        )
    
    with col2:
        manual_lon = st.number_input(
            "Longitude:",
            min_value=-180.0,
            max_value=180.0,
            value=selected_lon if selected_lon else 106.8456,
            format="%.6f",
            key="manual_lon"
        )
    
    # Use manual input if map wasn't clicked
    if not selected_lat or not selected_lon:
        selected_lat = manual_lat
        selected_lon = manual_lon
    
    st.info(f"🎯 Search coordinates: **{selected_lat:.6f}, {selected_lon:.6f}**")
    
    # Step 2: Additional filters
    st.markdown("**Step 2: Additional Filters (Optional)**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Kondisi Wilayah Sekitar:**")
        # Get unique values for kondisi_wilayah_sekitar
        with st.spinner("Loading kondisi wilayah options..."):
            kondisi_values, _ = db.get_column_unique_values(table, 'kondisi_wilayah_sekitar', schema)
        
        selected_kondisi = []
        if kondisi_values:
            selected_kondisi = st.multiselect(
                "Select kondisi wilayah:",
                kondisi_values,
                help="Filter by surrounding area conditions",
                key="point_kondisi"
            )
    
    with col2:
        st.markdown("**Luas Tanah (m²):**")
        # Get min/max for luas_tanah
        luas_query = f'SELECT MIN("luas_tanah") as min_val, MAX("luas_tanah") as max_val FROM "{schema}"."{table}" WHERE "luas_tanah" IS NOT NULL'
        luas_result, _ = db.execute_query(luas_query)
        
        if luas_result is not None and len(luas_result) > 0:
            min_luas = float(luas_result['min_val'].iloc[0])
            max_luas = float(luas_result['max_val'].iloc[0])
            
            luas_range = st.slider(
                "Luas tanah range:",
                min_value=min_luas,
                max_value=max_luas,
                value=(min_luas, max_luas),
                help="Filter by land area",
                key="point_luas"
            )
        else:
            luas_range = None
    
    with col3:
        st.markdown("**Lebar Jalan di Depan (m):**")
        # Get min/max for lebar_jalan_di_depan
        lebar_query = f'SELECT MIN("lebar_jalan_di_depan") as min_val, MAX("lebar_jalan_di_depan") as max_val FROM "{schema}"."{table}" WHERE "lebar_jalan_di_depan" IS NOT NULL'
        lebar_result, _ = db.execute_query(lebar_query)
        
        if lebar_result is not None and len(lebar_result) > 0:
            min_lebar = float(lebar_result['min_val'].iloc[0])
            max_lebar = float(lebar_result['max_val'].iloc[0])
            
            lebar_range = st.slider(
                "Lebar jalan range:",
                min_value=min_lebar,
                max_value=max_lebar,
                value=(min_lebar, max_lebar),
                help="Filter by road width in front",
                key="point_lebar"
            )
        else:
            lebar_range = None
    
    # Search button
    st.markdown("**Step 3: Execute Search**")
    if st.button("🔍 Search Nearest Properties", type="primary", use_container_width=True, key="point_search"):
        # Prepare filter parameters
        luas_tanah_range = luas_range if luas_range and luas_range != (min_luas, max_luas) else None
        lebar_jalan_range = lebar_range if lebar_range and lebar_range != (min_lebar, max_lebar) else None
        kondisi_wilayah_opt = selected_kondisi if selected_kondisi else None
        
        # Build and execute query
        search_properties_by_location(
            db, schema, table, 
            selected_lat, selected_lon,
            luas_tanah_range, lebar_jalan_range, kondisi_wilayah_opt
        )

def load_administrative_data(db, schema, table, filters):
    """Load data using administrative filtering"""
    try:
        # Get mandatory columns
        mandatory_cols = ['luas_tanah', 'hpm', 'longitude', 'latitude']
        
        # Build SELECT clause
        select_columns = ', '.join([f'"{col}"' for col in mandatory_cols])
        query = f'SELECT {select_columns} FROM "{schema}"."{table}"'
        
        # Build WHERE clause
        where_conditions = []
        
        for col, filter_val in filters.items():
            if isinstance(filter_val, list):
                if filter_val:
                    values_str = "', '".join([str(v) for v in filter_val])
                    where_conditions.append(f'"{col}" IN (\'{values_str}\')')
            elif isinstance(filter_val, dict):
                if filter_val.get('type') == 'range':
                    where_conditions.append(f'"{col}" BETWEEN {filter_val["min"]} AND {filter_val["max"]}')
        
        if where_conditions:
            query += ' WHERE ' + ' AND '.join(where_conditions)
        
        query += ' LIMIT 100000'
        
        # Execute query
        with st.spinner("Loading data..."):
            result_df, message = db.execute_query(query)
        
        if result_df is not None:
            st.session_state.current_data = result_df
            st.session_state.applied_filters = filters
            
            st.success(f"✅ Data loaded successfully! Retrieved {len(result_df):,} rows")
            st.dataframe(result_df.head(10), use_container_width=True)
            
            # Show summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(result_df):,}")
            with col2:
                st.metric("Avg HPM", f"{result_df['hpm'].mean():,.0f}")
            with col3:
                st.metric("Avg Luas", f"{result_df['luas_tanah'].mean():.0f} m²")
            with col4:
                st.metric("Data Points", len(result_df))
        else:
            st.error(f"Failed to load data: {message}")
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def search_properties_by_location(db, schema, table, lat, lon, luas_tanah_range, lebar_jalan_range, kondisi_wilayah_opt):
    """Search properties by location using PostGIS"""
    try:
        # Build SQL query
        sql_query = build_property_query(
            lat, lon,
            luas_tanah_range=luas_tanah_range,
            lebar_jalan_range=lebar_jalan_range,
            kondisi_wilayah_opt=kondisi_wilayah_opt,
            schema=schema,
            table=table
        )
        
        # Execute query
        with st.spinner("Searching nearest properties..."):
            result_df, message = db.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            st.session_state.current_data = result_df
            
            st.success(f"✅ Found {len(result_df)} properties near your location!")
            
            # Show results summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Properties Found", len(result_df))
            with col2:
                if 'distance_meters' in result_df.columns:
                    avg_distance = result_df['distance_meters'].mean()
                    st.metric("Avg Distance", f"{avg_distance/1000:.1f} km")
            with col3:
                if 'hpm' in result_df.columns:
                    avg_hpm = result_df['hpm'].mean()
                    st.metric("Avg HPM", f"{avg_hpm:,.0f}")
            with col4:
                if 'luas_tanah' in result_df.columns:
                    avg_luas = result_df['luas_tanah'].mean()
                    st.metric("Avg Luas", f"{avg_luas:.0f} m²")
            
            # Show data preview
            st.markdown("**Search Results Preview:**")
            display_columns = ['distance_meters', 'hpm', 'luas_tanah', 'lebar_jalan_di_depan', 'kondisi_wilayah_sekitar']
            available_columns = [col for col in display_columns if col in result_df.columns]
            
            if available_columns:
                preview_df = result_df[available_columns].head(10).copy()
                if 'distance_meters' in preview_df.columns:
                    preview_df['distance_km'] = preview_df['distance_meters'] / 1000
                    preview_df = preview_df.drop('distance_meters', axis=1)
                
                st.dataframe(preview_df, use_container_width=True)
            else:
                st.dataframe(result_df.head(10), use_container_width=True)
            
            # Show applied filters summary
            st.markdown("**Applied Search Filters:**")
            st.write(f"📍 **Search Center:** {lat:.6f}, {lon:.6f}")
            
            if luas_tanah_range:
                st.write(f"🏞️ **Luas Tanah:** {luas_tanah_range[0]:.0f} - {luas_tanah_range[1]:.0f} m²")
            
            if lebar_jalan_range:
                st.write(f"🛣️ **Lebar Jalan:** {lebar_jalan_range[0]:.0f} - {lebar_jalan_range[1]:.0f} m")
            
            if kondisi_wilayah_opt:
                st.write(f"🏘️ **Kondisi Wilayah:** {', '.join(kondisi_wilayah_opt)}")
            
            st.write(f"📊 **Result Limit:** Nearest 300 properties")
            
        else:
            st.warning("No properties found matching your criteria. Try adjusting your filters or location.")
            
    except Exception as e:
        st.error(f"Error searching properties: {str(e)}")
        st.code(sql_query)  # Show query for debugging


# Integration function to be added to your main render_data_selection() function
def integrate_land_market_filtering():
    """
    Replace the existing Land Market filtering section in render_data_selection() 
    with this enhanced version that provides two filtering options.
    
    In your main code, replace the land market filtering section with:
    
    if st.session_state.selected_table == 'engineered_property_data':
        render_land_market_filtering()
        return  # Exit early for land market data
    
    This should be placed right after the table selection and before the 
    existing column selection logic.
    """
    pass

def render_data_selection():
    """Render data selection and filtering section"""
    st.markdown('<div class="section-header">🎯 Pilih dan Filter Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.db_connection.connection_status:
        st.warning("⚠️ Please connect to database first")
        return
    
    # Available data tables
    available_tables = {
        'condo_converted_2025': {
            'name': 'Condo Data 2025',
            'icon': '🏢',
            'description': 'Condominium properties data for 2025',
            'color': '#3498db'
        },
        'hotel_converted_2025': {
            'name': 'Hotel Data 2025', 
            'icon': '🏨',
            'description': 'Hotel properties data for 2025',
            'color': '#e74c3c'
        },
        'hospital_converted_2025': {
            'name': 'Hospital Data 2025',
            'icon': '🏥',
            'description': 'Hospital properties data for 2025',
            'color': '#9b59b6'
        },
        'office_converted_2025': {
            'name': 'Office Data 2025',
            'icon': '🏢',
            'description': 'Office properties data for 2025', 
            'color': '#f39c12'
        },
        'retail_converted_2025': {
            'name': 'Retail Data 2025',
            'icon': '🏬',
            'description': 'Retail properties data for 2025',
            'color': '#27ae60'
        },
        'engineered_property_data': {
            'name': 'Land Market',
            'icon': '🏞️',
            'description': 'Land market data analysis',
            'color': '#34495e'
        }
    }

    # Table selection
    st.markdown("### 📋 **Pilih Data!**")
    st.markdown("Pilih salah satu data yang tersedia!")
    
    # Create clickable table cards
    cols = st.columns(3)
    
    for i, (table_key, table_info) in enumerate(available_tables.items()):
        with cols[i % 3]:
            # Create a container for the card
            card_container = st.container()
            
            # Check if this table is selected
            is_selected = st.session_state.selected_table == table_key
            card_class = "selected-table" if is_selected else "data-table-card"
            
            # Create the card with button
            if st.button(
                f"{table_info['icon']} {table_info['name']}\n{table_info['description']}", 
                key=f"table_{table_key}",
                use_container_width=True
            ):
                st.session_state.selected_table = table_key
                st.session_state.table_columns = None  # Reset columns when table changes
                st.session_state.applied_filters = {}  # Reset filters
                st.rerun()
    
    # Show selected table
    if st.session_state.selected_table:
        selected_info = available_tables[st.session_state.selected_table]
        st.markdown(f'<div class="success-box">✅ Selected: {selected_info["icon"]} {selected_info["name"]}</div>', unsafe_allow_html=True)
        
        # Get table columns
        if st.session_state.table_columns is None:
            with st.spinner("Loading table structure..."):
                columns_df, msg = st.session_state.db_connection.get_table_columns(
                    st.session_state.selected_table, 
                    st.session_state.get('schema', 'public')
                )
                if columns_df is not None:
                    st.session_state.table_columns = columns_df
                else:
                    st.error(f"Failed to load table structure: {msg}")
                    return
        
        # Column selection and filtering
        if st.session_state.table_columns is not None:
            st.markdown("### 📊 **Column Selection & Filtering**")
            
            # Show available columns
            with st.expander("📋 **Available Columns**", expanded=False):
                st.dataframe(st.session_state.table_columns, use_container_width=True)
            
            # Add data preview section
            st.markdown("### 👀 **Data Preview**")
            with st.spinner("Loading data preview..."):
                # Get sample data from the table
                schema = st.session_state.get('schema', 'public')
                preview_query = f'SELECT * FROM "{schema}"."{st.session_state.selected_table}" LIMIT 5'
                
                try:
                    preview_df, preview_msg = st.session_state.db_connection.execute_query(preview_query)
                    if preview_df is not None:
                        st.markdown("**First 5 rows of the selected table:**")
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Show basic info about the preview
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Preview Rows", len(preview_df))
                        with col2:
                            st.metric("Total Columns", len(preview_df.columns))
                        with col3:
                            numeric_cols = len(preview_df.select_dtypes(include=[np.number]).columns)
                            st.metric("Numeric Columns", numeric_cols)
                    else:
                        st.error(f"Failed to load preview: {preview_msg}")
                except Exception as e:
                    st.error(f"Error loading data preview: {str(e)}")
            
            st.markdown("---")
            
            # Column selection
            st.markdown("### 📊 **Column Selection**")
            
            available_columns = st.session_state.table_columns['column_name'].tolist()

            if st.session_state.selected_table == 'engineered_property_data':
                mandatory_cols = ['luas_tanah', 'hpm', 'longitude', 'latitude']
                user_selectable_cols = [col for col in available_columns if col not in mandatory_cols]

                selected_user_cols = st.multiselect(
                    "Select up to 6 columns for Land Market analysis:",
                    user_selectable_cols,
                    default=[],
                    help="Mandatory columns luas_tanah, hpm, longitude, latitude are always included."
                )

                if len(selected_user_cols) > 6:
                    st.warning("⚠️ You can select a maximum of 6 columns only. Please deselect extra columns.")
                    
                selected_columns = mandatory_cols + selected_user_cols

                st.info(f"📊 {len(selected_columns)} columns selected (4 mandatory + {len(selected_user_cols)} user-selected)")
   
            else:
                selected_columns = available_columns
                # Then continue with the flexible filter UI below as you currently have it
                st.markdown("Apply filters to focus on specific data subsets (you can apply multiple filters):")
                
            
            if selected_columns:
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown("### 🔍 **Data Selection**")

                db = st.session_state.db_connection
                schema = st.session_state.get('schema', 'public')
                table = st.session_state.selected_table

                if st.session_state.selected_table == 'engineered_property_data':
                    render_land_market_filtering()
                    return  # Exit early for land market data

                else:
                    # For other tables: only require province, others are optional
                    st.markdown("#### 🎯 **Required: Select Province/Region**")
                    with st.spinner("Loading province options..."):
                        province_values, province_msg = db.get_column_unique_values(table, 'wadmpr', schema)

                    if province_values:
                        selected_province = st.selectbox(
                            "Choose a Province/Region:",
                            [""] + province_values,
                            help="You must select a province/region to continue"
                        )
                        if not selected_province:
                            st.error("❌ Please select a province/region to continue")
                            st.stop()
                    else:
                        st.error(f"Could not load provinces: {province_msg}")
                        st.stop()

                    selected_regency = None
                    selected_district = None
                    selected_subdistrict = None

                    # Regency/City (optional)
                    st.markdown("#### Optional: Select Regency/City")
                    with st.spinner("Loading regency/city options..."):
                        regency_query = f'''
                            SELECT DISTINCT wadmkk FROM "{schema}"."{table}"
                            WHERE wadmpr = '{selected_province}'
                            ORDER BY wadmkk
                        '''
                        regency_df, _ = db.execute_query(regency_query)
                        regency_values = regency_df['wadmkk'].dropna().tolist() if regency_df is not None else []

                    selected_regency = st.selectbox(
                        "Choose a Regency/City (optional):",
                        [""] + regency_values
                    )
                    # if selected_regency:
                    #     filters['wadmkk'] = [selected_regency]

                    # District (optional, only if regency is selected)
                    if selected_regency:
                        st.markdown("#### Optional: Select District")
                        with st.spinner("Loading district options..."):
                            district_query = f'''
                                SELECT DISTINCT wadmkc FROM "{schema}"."{table}"
                                WHERE wadmpr = '{selected_province}' AND wadmkk = '{selected_regency}'
                                ORDER BY wadmkc
                            '''
                            district_df, _ = db.execute_query(district_query)
                            district_values = district_df['wadmkc'].dropna().tolist() if district_df is not None else []

                        selected_district = st.selectbox(
                            "Choose a District (optional):",
                            [""] + district_values
                        )
                        if selected_district:
                            filters['wadmkc'] = [selected_district]
                    else:
                        selected_district = None  # to be used in subdistrict query below

                    # Subdistrict (optional, only if district is selected)
                    if selected_district:
                        st.markdown("#### Optional: Select Subdistrict")
                        with st.spinner("Loading subdistrict options..."):
                            subdistrict_query = f'''
                                SELECT DISTINCT wadmkd FROM "{schema}"."{table}"
                                WHERE wadmpr = '{selected_province}' AND wadmkk = '{selected_regency}' AND wadmkc = '{selected_district}'
                                ORDER BY wadmkd
                            '''
                            subdistrict_df, _ = db.execute_query(subdistrict_query)
                            subdistrict_values = subdistrict_df['wadmkd'].dropna().tolist() if subdistrict_df is not None else []

                        selected_subdistrict = st.selectbox(
                            "Choose a Subdistrict (optional):",
                            [""] + subdistrict_values
                        )
                        if selected_subdistrict:
                            filters['wadmkd'] = [selected_subdistrict]

                # Initialize filters
                filters = {}
                filters['wadmpr'] = [selected_province]
                if selected_regency:  # avoid empty/null/None
                    filters['wadmkk'] = [selected_regency]
                if selected_district:
                    filters['wadmkc'] = [selected_district]
                if selected_subdistrict:
                    filters['wadmkd'] = [selected_subdistrict]


                # Now, you can safely continue with other steps, as district is guaranteed selected
                st.success("All required region filters selected! Continue to next steps.")

                # --- END: Region Filters ---

                # --- Block next steps unless District is selected ---
                if filters.get('wadmkc'):
                    st.success("✅ District selected")

                    # ---- Optional Filters Example ----
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Filter by Year:**")

                        # Build WHERE clause for years (by area)
                        area_where_parts = []
                        for key in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']:
                            if filters.get(key):
                                area_where_parts.append(f'"{key}" = \'{filters[key][0]}\'')
                        area_where_clause = " AND ".join(area_where_parts)
                        years_query = f"""
                            SELECT DISTINCT tahun_pengambilan_data 
                            FROM "{schema}"."{table}"
                            WHERE {area_where_clause}
                            ORDER BY tahun_pengambilan_data
                        """
                        years_df, _ = db.execute_query(years_query)
                        year_values = years_df['tahun_pengambilan_data'].dropna().tolist() if years_df is not None else []

                        if year_values:
                            selected_years = st.multiselect(
                                "Select years:",
                                year_values,
                                default=year_values,
                                help="Choose which years to include in analysis"
                            )
                            if len(selected_years) < len(year_values):
                                filters['tahun_pengambilan_data'] = selected_years


                        with col2:
                            st.markdown("**Filter by HPM (Price Range):**")
                            hpm_query = f"""
                                SELECT MIN("hpm") as min_hpm, MAX("hpm") as max_hpm 
                                FROM "{schema}"."{table}"
                                WHERE {area_where_clause} AND "hpm" IS NOT NULL
                            """
                            hpm_result, _ = db.execute_query(hpm_query)
                            if hpm_result is not None and len(hpm_result) > 0:
                                min_hpm = float(hpm_result['min_hpm'].iloc[0])
                                max_hpm = float(hpm_result['max_hpm'].iloc[0])

                                st.write(f"HPM range: {min_hpm:,.0f} - {max_hpm:,.0f}")

                                hpm_range = st.slider(
                                    "Select HPM range:",
                                    min_value=min_hpm,
                                    max_value=max_hpm,
                                    value=(min_hpm, max_hpm),
                                    step=1000.0,
                                    help="Filter properties by price per square meter"
                                )
                                if hpm_range != (min_hpm, max_hpm):
                                    filters['hpm'] = {'min': hpm_range[0], 'max': hpm_range[1], 'type': 'range'}

                
                else:
                    # Flexible filtering for other tables
                    st.markdown("Apply filters to focus on specific data subsets (you can apply multiple filters):")
                    
                    # Initialize session state for multiple filters if not exists
                    if 'filter_steps' not in st.session_state:
                        st.session_state.filter_steps = []
                    
                    filters = {}
                    
                    # Add filter button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_filter_column = st.selectbox(
                            "Add a new filter for column:",
                            ["Select a column..."] + selected_columns,
                            key="new_filter_selector"
                        )
                    with col2:
                        st.write("")
                        if st.button("➕ Add Filter", use_container_width=True) and new_filter_column != "Select a column...":
                            if new_filter_column not in [step['column'] for step in st.session_state.filter_steps]:
                                st.session_state.filter_steps.append({
                                    'column': new_filter_column,
                                    'id': len(st.session_state.filter_steps)
                                })
                                st.rerun()
                    
                    # Display and configure active filters
                    if st.session_state.filter_steps:
                        st.markdown("**Active Filters:**")
                        
                        for i, filter_step in enumerate(st.session_state.filter_steps):
                            filter_col = filter_step['column']
                            
                            with st.container():
                                # Filter header with remove button
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"**🔍 Filter {i+1}: `{filter_col}`**")
                                with col2:
                                    if st.button("❌", key=f"remove_filter_{i}", help="Remove this filter"):
                                        st.session_state.filter_steps.pop(i)
                                        st.rerun()
                                
                                # Get unique values for this column
                                with st.spinner(f"Loading values for {filter_col}..."):
                                    unique_values, msg = st.session_state.db_connection.get_column_unique_values(
                                        st.session_state.selected_table, 
                                        filter_col,
                                        st.session_state.get('schema', 'public')
                                    )
                                
                                if unique_values:
                                    # Determine filter type
                                    col_type = st.session_state.table_columns[
                                        st.session_state.table_columns['column_name'] == filter_col
                                    ]['data_type'].iloc[0]
                                    
                                    if len(unique_values) <= 50:  # Categorical filter
                                        selected_values = st.multiselect(
                                            f"Select values for {filter_col}:",
                                            unique_values,
                                            default=unique_values,
                                            key=f"filter_{filter_col}_{i}"
                                        )
                                        if len(selected_values) < len(unique_values):
                                            filters[filter_col] = selected_values
                                    
                                    else:  # Range or text filter
                                        if 'int' in col_type.lower() or 'float' in col_type.lower() or 'numeric' in col_type.lower():
                                            # Numeric range filter
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                min_val = st.number_input(
                                                    f"Min {filter_col}:", 
                                                    value=float(min(unique_values)), 
                                                    key=f"min_{filter_col}_{i}"
                                                )
                                            with col2:
                                                max_val = st.number_input(
                                                    f"Max {filter_col}:", 
                                                    value=float(max(unique_values)), 
                                                    key=f"max_{filter_col}_{i}"
                                                )
                                            
                                            if min_val != min(unique_values) or max_val != max(unique_values):
                                                filters[filter_col] = {'min': min_val, 'max': max_val, 'type': 'range'}
                                        
                                        else:
                                            # Text search filter
                                            search_term = st.text_input(
                                                f"Search in {filter_col}:", 
                                                key=f"search_{filter_col}_{i}",
                                                help="Use % as wildcard (e.g., 'Jakarta%' for starts with Jakarta)"
                                            )
                                            if search_term:
                                                filters[filter_col] = {'search': search_term, 'type': 'text'}
                                
                                else:
                                    st.warning(f"Could not load values for {filter_col}")
                                
                                st.markdown("---")
                        
                        # Clear all filters button
                        if st.button("🗑️ Clear All Filters", use_container_width=True):
                            st.session_state.filter_steps = []
                            st.rerun()
                    
                    else:
                        st.info("No filters applied. Use the dropdown above to add filters.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Get Data button
                st.markdown("### 🚀 **Load Data**")
                
                # col1, col2, col3 = st.columns([2, 1, 1])
                
                # with col1:
                if st.button("🎯 Get Data", type="primary", use_container_width=True):
                    # Build the SQL query
                    schema = st.session_state.get('schema', 'public')
                    
                    # Build SELECT clause
                    select_columns = ', '.join([f'"{col}"' for col in selected_columns])
                    query = f'SELECT {select_columns} FROM "{schema}"."{st.session_state.selected_table}"'
                    
                    # Build WHERE clause
                    where_conditions = []
                    
                    for col, filter_val in filters.items():
                        if isinstance(filter_val, list):  # Categorical filter
                            if filter_val:  # Only add condition if values are selected
                                values_str = "', '".join([str(v) for v in filter_val])
                                where_conditions.append(f'"{col}" IN (\'{values_str}\')')
                        
                        elif isinstance(filter_val, dict):
                            if filter_val.get('type') == 'range':
                                where_conditions.append(f'"{col}" BETWEEN {filter_val["min"]} AND {filter_val["max"]}')
                            elif filter_val.get('type') == 'text':
                                where_conditions.append(f'"{col}" ILIKE \'%{filter_val["search"]}%\'')
                    
                    if where_conditions:
                        query += ' WHERE ' + ' AND '.join(where_conditions)
                    
                    # Add limit for performance
                    query += ' LIMIT 100000'
                    
                    # Execute query
                    with st.spinner("Loading data..."):
                        result_df, message = st.session_state.db_connection.execute_query(query)
                    
                    if result_df is not None:
                        st.session_state.current_data = result_df
                        st.session_state.applied_filters = filters
                        
                        st.success(f"✅ Data loaded successfully! Retrieved {len(result_df):,} rows with {len(selected_columns)} columns.")
                        
                        # Show data preview
                        st.markdown("**Data Preview:**")
                        st.dataframe(result_df.head(10), use_container_width=True)
                        
                        # Show summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Rows", f"{len(result_df):,}")
                        with col2:
                            st.metric("Total Columns", len(result_df.columns))
                        with col3:
                            numeric_cols = len(result_df.select_dtypes(include=[np.number]).columns)
                            st.metric("Numeric Columns", numeric_cols)
                        with col4:
                            missing_pct = (result_df.isnull().sum().sum() / (result_df.shape[0] * result_df.shape[1]) * 100)
                            st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
                    
                    else:
                        st.error(f"Failed to load data: {message}")
                
                # with col2:
                st.write("")
                if st.button("🔄 Reset Filters", use_container_width=True):
                    st.session_state.applied_filters = {}
                    st.rerun()
                
                # with col3:
                st.write("")
                if st.button("📊 Show Query", use_container_width=True):
                    # Show the generated query
                    schema = st.session_state.get('schema', 'public')
                    select_columns = ', '.join([f'"{col}"' for col in selected_columns])
                    query = f'SELECT {select_columns} FROM "{schema}"."{st.session_state.selected_table}"'
                    
                    where_conditions = []
                    for col, filter_val in filters.items():
                        if isinstance(filter_val, list) and filter_val:
                            values_str = "', '".join([str(v) for v in filter_val])
                            where_conditions.append(f'"{col}" IN (\'{values_str}\')')
                        elif isinstance(filter_val, dict):
                            if filter_val.get('type') == 'range':
                                where_conditions.append(f'"{col}" BETWEEN {filter_val["min"]} AND {filter_val["max"]}')
                            elif filter_val.get('type') == 'text':
                                where_conditions.append(f'"{col}" ILIKE \'%{filter_val["search"]}%\'')
                    
                    if where_conditions:
                        query += ' WHERE ' + ' AND '.join(where_conditions)
                    
                    st.code(query, language="sql")
                
                # Show applied filters summary
                if filters:
                    st.markdown("**Applied Filters:**")
                    for col, filter_val in filters.items():
                        if isinstance(filter_val, list):
                            st.write(f"• `{col}`: {len(filter_val)} selected values")
                        elif isinstance(filter_val, dict):
                            if filter_val.get('type') == 'range':
                                st.write(f"• `{col}`: {filter_val['min']} - {filter_val['max']}")
                            elif filter_val.get('type') == 'text':
                                st.write(f"• `{col}`: contains '{filter_val['search']}'")

def render_data_chatbot():
    """Render Data Chatbot section"""
    st.markdown('<div class="section-header">💬 RHR AI</div>', unsafe_allow_html=True)
    
    # Check prerequisites
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load property data first using the Data Selection section")
        st.info("💡 Go to 'Pilih dan Filter Data' tab and load your dataset")
        return
    
    api_key = get_api_key("chatbot")
    if not api_key:
        st.warning("Please provide your OpenAI API key for the chatbot")
        return
    
    # Initialize session state for chatbot
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []
    
    if 'chatbot_system_prompt' not in st.session_state:
        st.session_state.chatbot_system_prompt = None
    
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
    
    df = st.session_state.current_data.copy()
    
    # Initialize chatbot
    try:
        chatbot = DataChatbot(api_key)
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return
    
    # Dataset overview
    st.markdown("### 📊 **Dataset Overview**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numerical", numeric_cols)
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Categorical", categorical_cols)
    with col5:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Data Quality", f"{100-missing_pct:.1f}%")
    
    # Data preview
    with st.expander("📋 **Data Preview**", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        # Quick stats
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            st.markdown("**Quick Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)
    
    # Initialize system prompt
    if not st.session_state.chatbot_initialized:
        with st.spinner("🧠 Initializing AI chatbot with your dataset..."):
            st.session_state.chatbot_system_prompt = chatbot.create_system_prompt(df)
            
            # Generate initial analysis
            try:
                initial_message = """
                Halo! Saya RHR AI. Saya siap membantu Anda memahami data Anda.
                Silakan tanyakan apa pun tentang data Anda - Saya dapat membantu dengan: 
                • Analisis dan wawasan statistik 
                • Pola dan tren data
                • Analisis pasar properti
                • Penilaian kualitas data 
                • Rekomendasi visualisasi 
                • Pertanyaan khusus tentang properti Anda
                Apa yang ingin Anda ketahui terlebih dahulu?
                """
                
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": initial_message
                })
                
                st.session_state.chatbot_initialized = True
                
            except Exception as e:
                st.error(f"Failed to initialize chatbot analysis: {str(e)}")
                return
    
    # Map visualization
    if any(col.lower() in ['lat', 'latitude'] for col in df.columns) and any(col.lower() in ['lon', 'lng', 'longitude'] for col in df.columns):
        st.markdown("### 🗺️ **Property Location Map**")
        
        # Find latitude and longitude columns
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            if col.lower() in ['lat', 'latitude'] and lat_col is None:
                lat_col = col
            elif col.lower() in ['lon', 'lng', 'longitude'] and lon_col is None:
                lon_col = col
        
        if lat_col and lon_col:
            # Clean and filter valid coordinates
            map_df = df[[lat_col, lon_col]].copy()
            map_df = map_df.dropna()
            
            # Convert to numeric first
            try:
                # Convert coordinates to numeric, coerce errors to NaN
                map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors='coerce')
                map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors='coerce')
                
                # Remove rows with invalid coordinates
                map_df = map_df.dropna()
                
                # Filter valid coordinate ranges
                map_df = map_df[
                    (map_df[lat_col] >= -90) & (map_df[lat_col] <= 90) &
                    (map_df[lon_col] >= -180) & (map_df[lon_col] <= 180)
                ]
            except Exception as e:
                st.error(f"Error processing coordinates: {str(e)}")
                map_df = pd.DataFrame()  # Empty dataframe if conversion fails
            
            if not map_df.empty:
                # Add price or value column if available for color coding
                value_col = None
                if 'hpm' in df.columns:
                    value_col = 'hpm'
                elif 'price' in df.columns:
                    value_col = 'price'
                elif 'harga' in df.columns:
                    value_col = 'harga'
                
                if value_col:
                    map_df[value_col] = df[value_col]
                    map_df = map_df.dropna()
                
                # Create the map
                if len(map_df) > 0:
                    fig = go.Figure()
                    
                    if value_col and value_col in map_df.columns:
                        # Colored markers based on value
                        fig.add_trace(go.Scattermapbox(
                            lat=map_df[lat_col],
                            lon=map_df[lon_col],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=map_df[value_col],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title=value_col)
                            ),
                            text=[f"{value_col}: {val:,.0f}" for val in map_df[value_col]],
                            hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
                            name='Properties'
                        ))
                    else:
                        # Simple markers
                        fig.add_trace(go.Scattermapbox(
                            lat=map_df[lat_col],
                            lon=map_df[lon_col],
                            mode='markers',
                            marker=dict(size=8, color='blue'),
                            hovertemplate='Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
                            name='Properties'
                        ))
                    
                    # Map layout
                    center_lat = map_df[lat_col].mean()
                    center_lon = map_df[lon_col].mean()
                    
                    fig.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=10
                        ),
                        height=500,
                        margin=dict(l=0, r=0, t=0, b=0),
                        title=f"Property Locations ({len(map_df)} properties)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # # Map statistics
                    # col1, col2, col3 = st.columns(3)
                    # with col1:
                    #     st.metric("Properties Mapped", len(map_df))
                    # with col2:
                    #     st.metric("Center Latitude", f"{center_lat:.4f}")
                    # with col3:
                    #     st.metric("Center Longitude", f"{center_lon:.4f}")
                else:
                    st.info("No valid coordinates found for mapping")
            else:
                st.warning("No valid coordinate data available for mapping")
        else:
            st.info("Latitude/Longitude columns not detected in the dataset")
    else:
        st.info("Geographic data not available - add latitude/longitude columns to see property locations")
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 **RHR AI**")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chatbot_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            try:
                # Build message history for context
                messages = [SystemMessage(content=st.session_state.chatbot_system_prompt)]
                
                for msg in st.session_state.chatbot_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                # Stream response
                response_container = st.empty()
                full_response = ""
                
                try:
                    for chunk in chatbot.llm.stream(messages):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            response_container.markdown(full_response + "▌")
                        
                    response_container.markdown(full_response)
                    
                except Exception as stream_error:
                    # Fallback to non-streaming
                    response = chatbot.llm.invoke(messages)
                    full_response = response.content
                    response_container.markdown(full_response)
                
                # Add assistant response to history
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": full_response
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": "I apologize, but I encountered an error. Please try rephrasing your question or ask something else about your data."
                })
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("### ⚡ **Quick Analysis Options**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Data Summary", use_container_width=True):
            summary_prompt = "Give me a comprehensive summary of this dataset including key statistics, patterns, and insights."
            st.session_state.chatbot_messages.append({"role": "user", "content": summary_prompt})
            try:
                messages = [SystemMessage(content=st.session_state.chatbot_system_prompt)]
                for msg in st.session_state.chatbot_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                response = chatbot.llm.invoke(messages)
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    with col2:
        if st.button("🏠 Property Insights", use_container_width=True):
            property_prompt = "Analyze the property data and provide key insights about pricing, locations, and market trends."
            st.session_state.chatbot_messages.append({"role": "user", "content": property_prompt})
            try:
                messages = [SystemMessage(content=st.session_state.chatbot_system_prompt)]
                for msg in st.session_state.chatbot_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                response = chatbot.llm.invoke(messages)
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    with col3:
        if st.button("📈 Price Analysis", use_container_width=True):
            price_prompt = "Analyze the pricing patterns in this dataset. What are the price ranges, averages, and any notable pricing trends?"
            st.session_state.chatbot_messages.append({"role": "user", "content": price_prompt})
            try:
                messages = [SystemMessage(content=st.session_state.chatbot_system_prompt)]
                for msg in st.session_state.chatbot_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                response = chatbot.llm.invoke(messages)
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    with col4:
        if st.button("🔍 Data Quality", use_container_width=True):
            quality_prompt = "Assess the data quality of this dataset. What are the missing values, outliers, and data quality issues?"
            st.session_state.chatbot_messages.append({"role": "user", "content": quality_prompt})
            try:
                messages = [SystemMessage(content=st.session_state.chatbot_system_prompt)]
                for msg in st.session_state.chatbot_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                response = chatbot.llm.invoke(messages)
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Chat management
    st.markdown("---")
    st.markdown("### 🛠️ **Chat Management**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.session_state.chatbot_initialized = False
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "dataset_shape": df.shape,
                "selected_table": st.session_state.selected_table,
                "applied_filters": st.session_state.applied_filters,
                "chat_messages": st.session_state.chatbot_messages
            }
            
            st.download_button(
                label="📄 Download Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"data_chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Sidebar info for chatbot
    if len(st.session_state.chatbot_messages) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**💬 Chat Status**")
        st.sidebar.success(f"✅ {len(st.session_state.chatbot_messages)} messages")
        st.sidebar.info(f"🤖 AI Model: gpt-4.1-mini")
        st.sidebar.info(f"📊 Dataset: {df.shape[0]:,} rows")

@st.cache_resource
def get_pyg_renderer(df: pd.DataFrame, spec_path: str) -> "StreamlitRenderer":
    return StreamlitRenderer(df, spec=spec_path, spec_io_mode="rw")

def render_dashboard():
    st.markdown('<div class="section-header">📊 Dashboard</div>', unsafe_allow_html=True)
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load property data first using the Data Selection section")
        st.info("💡 Go to 'Pilih dan Filter Data' tab and load your dataset")
        return

    df = st.session_state.current_data.copy()
    spec_path = f"./pyg_config_{st.session_state.selected_table or 'default'}.json"  # one config file per table
    pyg_app = get_pyg_renderer(df, spec_path)
    pyg_app.explorer()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header"> RHR Market Research Agent</h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <h1 style="display: flex; align-items: center;">
            <img src="https://kjpp.rhr.co.id/wp-content/uploads/2020/12/LOGO_KJPP_RHR_1_resize.png" 
                alt="Logo" style="height:48px; margin-right: 20px;">
            <span style="font-weight: bold; font-size: 1.5rem;"></span>
        </h1>
        """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    sections = [
        "🔗 Koneksi Database",
        "🎯 Pilih dan Filter Data",
        "💬 RHR AI",
        "📊 Dashboard"
    ]
    
    selected_section = st.sidebar.radio("Go to:", sections)
    
    # Render selected section
    if selected_section == "🔗 Koneksi Database":
        render_database_connection()
    
    elif selected_section == "🎯 Pilih dan Filter Data":
        render_data_selection()
    
    elif selected_section == "💬 RHR AI":
        render_data_chatbot()
    
    elif selected_section == "📊 Dashboard":
        render_dashboard()
    
    # Sidebar info and status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 System Status**")
    
    # Connection status
    if st.session_state.db_connection.connection_status:
        st.sidebar.success("✅ Database Connected")
    else:
        st.sidebar.error("❌ Database Disconnected")
    
    # Table selection status
    if st.session_state.selected_table:
        st.sidebar.success(f"✅ Table: {st.session_state.selected_table}")
    else:
        st.sidebar.warning("⚠️ No Table Selected")
    
    # Data status
    if st.session_state.current_data is not None:
        st.sidebar.success(f"✅ Data: {st.session_state.current_data.shape[0]:,} rows")
        st.sidebar.info(f"📊 Columns: {st.session_state.current_data.shape[1]}")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")
    
    # Filter status
    if st.session_state.applied_filters:
        st.sidebar.success(f"🔍 Filters: {len(st.session_state.applied_filters)} applied")
    
    # Chatbot status
    if 'chatbot_messages' in st.session_state and len(st.session_state.chatbot_messages) > 0:
        st.sidebar.success(f"💬 Chat: {len(st.session_state.chatbot_messages)} messages")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ℹ️ About**")
    st.sidebar.info(
        """
        **RHR Market Research Agent**
        
        🏠 Advanced property analysis with live market data
        🤖 AI-powered insights and predictions  
        📊 Professional statistical modeling
        📈 Interactive visualizations
        
        Built with Streamlit, OpenAI
        """
    )
    
    # Sidebar dataset info
    if st.session_state.selected_table:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📋 Current Dataset**")
        
        table_info = {
            'condo_converted_2025': {
                'name': 'Condo 2025',
                'icon': '🏢'
            },
            'hotel_converted_2025': {
                'name': 'Hotel 2025', 
                'icon': '🏨'
            },
            'office_converted_2025': {
                'name': 'Office 2025',
                'icon': '🏢'
            },
            'retail_converted_2025': {
                'name': 'Retail 2025',
                'icon': '🏬'
            },
            'hospital_converted_2025': {
                'name': 'Hospital 2025',
                'icon': '🏥'
            },
            'engineered_property_data': {
                'name': 'Land Market',
                'icon': '🏞️'
            }
        }
        
        current_table = table_info.get(st.session_state.selected_table, {'name': st.session_state.selected_table, 'icon': '📊'})
        
        # Special handling for engineered_property_data display name
        if st.session_state.selected_table == 'engineered_property_data':
            display_name = "Land Market"
        else:
            display_name = current_table['name']
        
        st.sidebar.info(f"{current_table['icon']} {display_name}")
        
        if st.session_state.current_data is not None:
            st.sidebar.metric("Records", f"{len(st.session_state.current_data):,}")
            st.sidebar.metric("Columns", f"{len(st.session_state.current_data.columns)}")
            
            # Memory usage
            memory_mb = st.session_state.current_data.memory_usage(deep=True).sum() / 1024**2
            st.sidebar.metric("Memory", f"{memory_mb:.1f} MB")
    
    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ Quick Actions**")
    
    if st.sidebar.button("🔄 Reset All"):
        # Reset all session state
        for key in ['current_data', 'selected_table', 'table_columns', 'applied_filters', 'chatbot_messages', 'chatbot_initialized']:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("All data reset!")
        st.rerun()
    
    if st.sidebar.button("💾 Export Session"):
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'selected_table': st.session_state.selected_table,
            'data_shape': st.session_state.current_data.shape if st.session_state.current_data is not None else None,
            'applied_filters': st.session_state.applied_filters,
            'has_chat_history': len(st.session_state.get('chatbot_messages', [])) > 0,
            'total_messages': len(st.session_state.get('chatbot_messages', []))
        }
        
        st.sidebar.download_button(
            label="📄 Download Session Info",
            data=json.dumps(session_data, indent=2),
            file_name=f"session_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    # Footer
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("""
    #     <h1 style="display: flex; align-items: center;">
    #         <img src="https://kjpp.rhr.co.id/wp-content/uploads/2020/12/LOGO_KJPP_RHR_1_resize.png" 
    #             alt="Logo" style="height:48px; margin-right: 20px;">
    #         <span style="font-weight: bold; font-size: 1.5rem;"></span>
    #     </h1>
    #     """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()