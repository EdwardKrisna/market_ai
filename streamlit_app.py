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
    cols = st.columns(3)
    
    for i, (table_key, table_info) in enumerate(available_tables.items()):
        with cols[i % 3]:
            if st.button(
                f"{table_info['icon']} {table_info['name']}\n{table_info['description']}", 
                key=f"table_{table_key}",
                use_container_width=True
            ):
                st.session_state.selected_table = table_key
                st.session_state.table_columns = None
                st.session_state.applied_filters = {}
                if 'selected_coordinates' in st.session_state:
                    del st.session_state.selected_coordinates
                st.rerun()
    
    # Show selected table and load data
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
        
        # Column selection
        available_columns = st.session_state.table_columns['column_name'].tolist()
        
        if st.session_state.selected_table == 'engineered_property_data':
            # Land Market specific setup
            mandatory_cols = ['luas_tanah', 'hpm', 'longitude', 'latitude']
            user_selectable_cols = [col for col in available_columns if col not in mandatory_cols]

            selected_user_cols = st.multiselect(
                "Select up to 6 columns for Land Market analysis:",
                user_selectable_cols,
                default=[],
                help="Mandatory columns luas_tanah, hmp, longitude, latitude are always included."
            )

            if len(selected_user_cols) > 6:
                st.warning("⚠️ You can select a maximum of 6 columns only.")
                
            selected_columns = mandatory_cols + selected_user_cols
            st.info(f"📊 {len(selected_columns)} columns selected (4 mandatory + {len(selected_user_cols)} user-selected)")
            
            # Filtering method selection
            st.markdown("### 🔍 **Choose Filtering Method**")
            filter_method = st.radio(
                "Select filtering approach:",
                ["Option 1: Administrative Area Filter", "Option 2: Map Location Filter"],
                help="Choose how you want to filter the land market data"
            )
            
            filters = {}
            db = st.session_state.db_connection
            schema = st.session_state.get('schema', 'public')
            table = st.session_state.selected_table
            
            if filter_method == "Option 1: Administrative Area Filter":
                # Administrative filtering
                with st.spinner("Loading administrative options..."):
                    province_values, _ = db.get_column_unique_values(table, 'wadmpr', schema)

                if province_values:
                    selected_province = st.selectbox("Province:", [""] + province_values)
                    if not selected_province:
                        st.error("❌ Please select a province")
                        return

                    regency_query = f"SELECT DISTINCT wadmkk FROM \"{schema}\".\"{table}\" WHERE wadmpr = '{selected_province}' ORDER BY wadmkk"
                    regency_df, _ = db.execute_query(regency_query)
                    regency_values = regency_df['wadmkk'].dropna().tolist() if regency_df is not None else []

                    if regency_values:
                        selected_regency = st.selectbox("Regency/City:", [""] + regency_values)
                        if not selected_regency:
                            st.error("❌ Please select a regency/city")
                            return

                        district_query = f"SELECT DISTINCT wadmkc FROM \"{schema}\".\"{table}\" WHERE wadmpr = '{selected_province}' AND wadmkk = '{selected_regency}' ORDER BY wadmkc"
                        district_df, _ = db.execute_query(district_query)
                        district_values = district_df['wadmkc'].dropna().tolist() if district_df is not None else []

                        if district_values:
                            selected_district = st.selectbox("District:", [""] + district_values)
                            if not selected_district:
                                st.error("❌ Please select a district")
                                return

                            filters['wadmpr'] = [selected_province]
                            filters['wadmkk'] = [selected_regency]
                            filters['wadmkc'] = [selected_district]
            
            elif filter_method == "Option 2: Map Location Filter":
                # Map-based filtering
                st.markdown("**Click on the map to select a location:**")
                
                # Simple HTML map with click functionality
                map_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
                    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                    <style>
                        #map { height: 400px; width: 100%; }
                        .coordinates { 
                            padding: 10px; background: #f0f0f0; margin: 10px 0; 
                            border-radius: 5px; text-align: center; font-weight: bold;
                        }
                    </style>
                </head>
                <body>
                    <div id="map"></div>
                    <div id="coordinates" class="coordinates">Click on the map to select coordinates</div>
                    
                    <script>
                        var map = L.map('map').setView([-2.5, 118.0], 6);
                        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
                        
                        var marker = null;
                        
                        map.on('click', function(e) {
                            var lat = e.latlng.lat;
                            var lon = e.latlng.lng;
                            
                            if (marker) map.removeLayer(marker);
                            marker = L.marker([lat, lon]).addTo(map);
                            
                            document.getElementById('coordinates').innerHTML = 
                                'Selected: ' + lat.toFixed(6) + ', ' + lon.toFixed(6);
                            
                            // Send coordinates to parent window
                            window.parent.postMessage({
                                type: 'mapClick',
                                lat: lat,
                                lon: lon
                            }, '*');
                        });
                    </script>
                </body>
                </html>
                """
                
                # Display map
                st.components.v1.html(map_html, height=500)
                
                # Initialize coordinates in session state
                if 'map_lat' not in st.session_state:
                    st.session_state.map_lat = 0.0
                if 'map_lon' not in st.session_state:
                    st.session_state.map_lon = 0.0
                
                # Coordinate inputs and validation
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    map_lat = st.number_input(
                        "Selected Latitude:", 
                        value=st.session_state.map_lat,
                        step=0.000001,
                        format="%.6f",
                        key="map_lat_input"
                    )
                
                with col2:
                    map_lon = st.number_input(
                        "Selected Longitude:", 
                        value=st.session_state.map_lon,
                        step=0.000001,
                        format="%.6f", 
                        key="map_lon_input"
                    )
                
                with col3:
                    st.write("")
                    if st.button("🔍 Use Point", use_container_width=True):
                        if map_lat != 0.0 and map_lon != 0.0:
                            try:
                                with st.spinner("Validating location..."):
                                    # Find nearest regency with data
                                    location_query = f"""
                                        SELECT DISTINCT wadmkk, wadmpr, COUNT(*) as data_count
                                        FROM "{schema}"."{table}"
                                        WHERE "latitude" IS NOT NULL AND "longitude" IS NOT NULL
                                        GROUP BY wadmkk, wadmpr
                                        ORDER BY ST_Distance(
                                            ST_Point(AVG(CAST("longitude" AS FLOAT)), AVG(CAST("latitude" AS FLOAT))),
                                            ST_Point({map_lon}, {map_lat})
                                        )
                                        LIMIT 1
                                    """
                                    location_result, _ = db.execute_query(location_query)
                                    
                                    if location_result is not None and len(location_result) > 0:
                                        regency = location_result['wadmkk'].iloc[0]
                                        province = location_result['wadmpr'].iloc[0]
                                        data_count = location_result['data_count'].iloc[0]
                                        
                                        # Validate data exists
                                        validation_query = f"SELECT COUNT(*) as count FROM \"{schema}\".\"{table}\" WHERE wadmkk = '{regency}'"
                                        validation_result, _ = db.execute_query(validation_query)
                                        
                                        if validation_result is not None and validation_result['count'].iloc[0] > 0:
                                            st.success(f"✅ Location validated! Area: {regency}, {province}")
                                            st.info(f"📊 Available data: {data_count:,} points")
                                            
                                            filters['location_search'] = {
                                                'lat': map_lat, 
                                                'lon': map_lon,
                                                'regency': regency
                                            }
                                        else:
                                            st.error("❌ No data for this area. Please select a different location.")
                                    else:
                                        st.error("❌ Could not find nearby areas. Please select a location in Indonesia.")
                                        
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                        else:
                            st.warning("⚠️ Please click on the map first.")
                
                # Show optional filters if location is validated
                if 'location_search' in filters:
                    regency = filters['location_search']['regency']
                    st.markdown("#### 🔍 **Optional Filters**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Kondisi Wilayah Sekitar
                    with col1:
                        kondisi_query = f"SELECT DISTINCT kondisi_wilayah_sekitar FROM \"{schema}\".\"{table}\" WHERE kondisi_wilayah_sekitar IS NOT NULL AND wadmkk = '{regency}' ORDER BY kondisi_wilayah_sekitar"
                        kondisi_df, _ = db.execute_query(kondisi_query)
                        kondisi_values = kondisi_df['kondisi_wilayah_sekitar'].tolist() if kondisi_df is not None else []
                        
                        if kondisi_values:
                            selected_kondisi = st.multiselect("Kondisi Wilayah:", kondisi_values, default=kondisi_values)
                            if len(selected_kondisi) < len(kondisi_values):
                                filters['kondisi_wilayah_sekitar'] = selected_kondisi
                    
                    # Luas Tanah
                    with col2:
                        luas_query = f"SELECT MIN(CAST(\"luas_tanah\" AS FLOAT)) as min_luas, MAX(CAST(\"luas_tanah\" AS FLOAT)) as max_luas FROM \"{schema}\".\"{table}\" WHERE \"luas_tanah\" IS NOT NULL AND CAST(\"luas_tanah\" AS FLOAT) > 0 AND wadmkk = '{regency}'"
                        luas_result, _ = db.execute_query(luas_query)
                        if luas_result is not None and len(luas_result) > 0:
                            min_luas = float(luas_result['min_luas'].iloc[0])
                            max_luas = float(luas_result['max_luas'].iloc[0])
                            
                            luas_range = st.slider("Luas Tanah (m²):", min_value=min_luas, max_value=max_luas, value=(min_luas, max_luas), step=10.0)
                            if luas_range != (min_luas, max_luas):
                                filters['luas_tanah'] = {'min': luas_range[0], 'max': luas_range[1], 'type': 'range'}
                    
                    # HPM
                    with col3:
                        hpm_query = f"SELECT MIN(CAST(\"hpm\" AS FLOAT)) as min_hpm, MAX(CAST(\"hpm\" AS FLOAT)) as max_hpm FROM \"{schema}\".\"{table}\" WHERE \"hpm\" IS NOT NULL AND CAST(\"hpm\" AS FLOAT) > 0 AND wadmkk = '{regency}'"
                        hpm_result, _ = db.execute_query(hpm_query)
                        if hpm_result is not None and len(hpm_result) > 0:
                            min_hpm = float(hpm_result['min_hpm'].iloc[0])
                            max_hpm = float(hpm_result['max_hpm'].iloc[0])
                            
                            hpm_range = st.slider("HPM (Price/m²):", min_value=min_hpm, max_value=max_hpm, value=(min_hpm, max_hpm), step=1000.0)
                            if hpm_range != (min_hpm, max_hpm):
                                filters['hpm'] = {'min': hpm_range[0], 'max': hpm_range[1], 'type': 'range'}

        else:
            # Other tables - simple province selection
            selected_columns = available_columns
            filters = {}
            
            db = st.session_state.db_connection
            schema = st.session_state.get('schema', 'public')
            table = st.session_state.selected_table
            
            province_values, _ = db.get_column_unique_values(table, 'wadmpr', schema)
            if province_values:
                selected_province = st.selectbox("Select Province:", [""] + province_values)
                if selected_province:
                    filters['wadmpr'] = [selected_province]

        # Load Data button and execution
        st.markdown("### 🚀 **Load Data**")
        
        if st.button("🎯 Get Data", type="primary", use_container_width=True):
            if not filters:
                st.warning("⚠️ Please apply at least one filter")
                return
                
            # Build query
            schema = st.session_state.get('schema', 'public')
            select_columns_str = ', '.join([f'"{col}"' for col in selected_columns])
            
            # Special handling for map-based search
            if 'location_search' in filters:
                location_info = filters['location_search']
                user_lat = location_info['lat']
                user_lon = location_info['lon']
                user_regency = location_info['regency']
                
                query = f"""
                SELECT {select_columns_str},
                       ST_Distance(
                           ST_Point(CAST("longitude" AS FLOAT), CAST("latitude" AS FLOAT)),
                           ST_Point({user_lon}, {user_lat})
                       ) as distance_meters
                FROM "{schema}"."{table}"
                WHERE "latitude" IS NOT NULL 
                AND "longitude" IS NOT NULL
                AND "wadmkk" = '{user_regency}'
                """
                
                # Add optional filters
                if 'kondisi_wilayah_sekitar' in filters:
                    kondisi_values = "', '".join(filters['kondisi_wilayah_sekitar'])
                    query += f" AND \"kondisi_wilayah_sekitar\" IN ('{kondisi_values}')"
                
                if 'luas_tanah' in filters and filters['luas_tanah'].get('type') == 'range':
                    luas_filter = filters['luas_tanah']
                    query += f" AND CAST(\"luas_tanah\" AS FLOAT) BETWEEN {luas_filter['min']} AND {luas_filter['max']}"
                
                if 'hpm' in filters and filters['hpm'].get('type') == 'range':
                    hpm_filter = filters['hpm']
                    query += f" AND CAST(\"hpm\" AS FLOAT) BETWEEN {hpm_filter['min']} AND {hpm_filter['max']}"
                
                query += " ORDER BY distance_meters ASC LIMIT 300"
            
            else:
                # Standard query
                query = f'SELECT {select_columns_str} FROM "{schema}"."{table}"'
                
                where_conditions = []
                for col, filter_val in filters.items():
                    if isinstance(filter_val, list) and filter_val:
                        values_str = "', '".join([str(v) for v in filter_val])
                        where_conditions.append(f'"{col}" IN (\'{values_str}\')')
                
                if where_conditions:
                    query += ' WHERE ' + ' AND '.join(where_conditions)
                
                query += ' LIMIT 100000'
            
            # Execute query
            with st.spinner("Loading data..."):
                result_df, message = st.session_state.db_connection.execute_query(query)
            
            if result_df is not None:
                st.session_state.current_data = result_df
                st.session_state.applied_filters = filters
                
                if 'location_search' in filters:
                    st.success(f"✅ Found {len(result_df):,} nearest properties (limited to 300)!")
                    if 'distance_meters' in result_df.columns:
                        min_dist = result_df['distance_meters'].min()
                        max_dist = result_df['distance_meters'].max()
                        st.info(f"📏 Distance range: {min_dist:.0f}m - {max_dist:.0f}m")
                else:
                    st.success(f"✅ Data loaded! Retrieved {len(result_df):,} rows with {len(selected_columns)} columns.")
                
                # Show preview
                st.dataframe(result_df.head(5), use_container_width=True)
            else:
                st.error(f"Failed to load data: {message}")

        # Reset and utility buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.applied_filters = {}
                if 'selected_coordinates' in st.session_state:
                    del st.session_state.selected_coordinates
                st.rerun()
        
        with col2:
            if st.button("📊 Show Query", use_container_width=True):
                st.info("Query will be shown here when Get Data is clicked")

    # JavaScript to handle map clicks and update input fields
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'mapClick') {
            // Update session state via a hidden form submission or other method
            console.log('Map clicked:', event.data.lat, event.data.lon);
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
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