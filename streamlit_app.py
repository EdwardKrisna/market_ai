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

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Real Estate Market Intelligence Platform",
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
            max_tokens=1000,
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
            st.success("✅ Using OpenAI API key from secrets.toml")
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
    st.markdown('<div class="section-header">🔗 Database Connection</div>', unsafe_allow_html=True)
    
    # Get connection parameters from secrets and auto-connect
    try:
        db_user = st.secrets["database"]["user"]
        db_pass = st.secrets["database"]["password"]
        db_host = st.secrets["database"]["host"]
        db_port = st.secrets["database"]["port"]
        db_name = st.secrets["database"]["name"]
        schema = st.secrets["database"]["schema"]
        
        st.success("✅ Database configuration loaded from secrets")
        
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
        st.markdown('<div class="success-box">🟢 Database Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">🔴 No Database Connection</div>', unsafe_allow_html=True)

def render_data_selection():
    """Render data selection and filtering section"""
    st.markdown('<div class="section-header">🎯 Data Selection & Filtering</div>', unsafe_allow_html=True)
    
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
    st.markdown("### 📋 **Select Data Table**")
    st.markdown("Choose one of the available property datasets:")
    
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
            
            # Special handling for Land Market (engineered_property_data)
            if st.session_state.selected_table == 'engineered_property_data':
                # For Land Market, only show specific columns
                land_columns = ['wadmpr','wadmkk','wadmkc','wadmkd','tahun_pengambilan_data','luas_tanah','kondisi_wilayah_sekitar','hpm', 'longitude', 'latitude']
                available_columns = [col for col in land_columns if col in available_columns]
                
                selected_columns = st.multiselect(
                    "Select columns for Land Market analysis (remove unwanted ones):",
                    available_columns,
                    default=available_columns,  # All Land Market columns selected by default
                    help="All required Land Market columns are selected. Remove any you don't need for analysis."
                )
                
                # Show column count
                st.info(f"📊 {len(selected_columns)} of {len(available_columns)} Land Market columns selected")
                
            else:
                # For other tables, show all columns with all selected by default
                selected_columns = st.multiselect(
                    "Select columns for analysis (remove unwanted ones):",
                    available_columns,
                    default=available_columns,  # All columns selected by default
                    help="All columns are selected by default. Remove the ones you don't need for your analysis."
                )
                
                # Show column count and quick info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 {len(selected_columns)} of {len(available_columns)} columns selected")
                with col2:
                    if len(selected_columns) != len(available_columns):
                        removed_count = len(available_columns) - len(selected_columns)
                        st.info(f"🗑️ {removed_count} columns removed")
                    else:
                        st.info("✅ All columns included")
            
            # Data filtering section
            if selected_columns:
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown("### 🔍 **Data Filtering (Optional)**")
                
                # Special handling for Land Market
                if st.session_state.selected_table == 'engineered_property_data':
                    st.warning("⚠️ **Land Market Requirements:** You must select exactly one wadmkc to continue (to prevent excessive data loading)")
                    
                    # wadmkc filter (required for Land Market)
                    st.markdown("#### 🎯 **Required: Select wadmkc**")
                    with st.spinner("Loading wadmkc options..."):
                        wadmkc_values, wadmkc_msg = st.session_state.db_connection.get_column_unique_values(
                            st.session_state.selected_table, 
                            'wadmkc',
                            st.session_state.get('schema', 'public')
                        )
                    
                    if wadmkc_values:
                        selected_wadmkc = st.selectbox(
                            "Choose one wadmkc:",
                            [""] + wadmkc_values,
                            help="You must select exactly one wadmkc area for analysis"
                        )
                        
                        if not selected_wadmkc:
                            st.error("❌ Please select a wadmkc to continue")
                            return
                        
                        # Initialize filters for Land Market
                        filters = {'wadmkc': [selected_wadmkc]}
                        
                        # Optional filters for Land Market
                        st.markdown("#### 🔧 **Optional Filters**")
                        
                        # Year filter
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Filter by Year:**")
                            with st.spinner("Loading years..."):
                                year_values, year_msg = st.session_state.db_connection.get_column_unique_values(
                                    st.session_state.selected_table, 
                                    'tahun_pengambilan_data',
                                    st.session_state.get('schema', 'public')
                                )
                            
                            if year_values:
                                selected_years = st.multiselect(
                                    "Select years:",
                                    year_values,
                                    default=year_values,
                                    help="Choose which years to include in analysis"
                                )
                                if len(selected_years) < len(year_values):
                                    filters['tahun_pengambilan_data'] = selected_years
                        
                        # HPM price range filter
                        with col2:
                            st.markdown("**Filter by HPM (Price Range):**")
                            with st.spinner("Loading HPM range..."):
                                hpm_query = f"""
                                SELECT MIN("hpm") as min_hpm, MAX("hpm") as max_hpm 
                                FROM "{st.session_state.get('schema', 'public')}"."{st.session_state.selected_table}"
                                WHERE "wadmkc" = '{selected_wadmkc}' AND "hpm" IS NOT NULL
                                """
                                hpm_result, hpm_msg = st.session_state.db_connection.execute_query(hpm_query)
                            
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
                        st.error(f"Could not load wadmkc values: {wadmkc_msg}")
                        return
                
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
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
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
                
                with col2:
                    st.write("")
                    if st.button("🔄 Reset Filters", use_container_width=True):
                        st.session_state.applied_filters = {}
                        st.rerun()
                
                with col3:
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
    st.markdown('<div class="section-header">💬 Chat with AI</div>', unsafe_allow_html=True)
    
    # Check prerequisites
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load property data first using the Data Selection section")
        st.info("💡 Go to 'Data Selection & Filtering' tab and load your dataset")
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
                    
                    # Map statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Properties Mapped", len(map_df))
                    with col2:
                        st.metric("Center Latitude", f"{center_lat:.4f}")
                    with col3:
                        st.metric("Center Longitude", f"{center_lon:.4f}")
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

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header">🏠 Real Estate Market Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    sections = [
        "🔗 Database Connection",
        "🎯 Data Selection & Filtering",
        "💬 Chat with AI"
    ]
    
    selected_section = st.sidebar.radio("Go to:", sections)
    
    # Render selected section
    if selected_section == "🔗 Database Connection":
        render_database_connection()
    
    elif selected_section == "🎯 Data Selection & Filtering":
        render_data_selection()
    
    elif selected_section == "💬 Chat with AI":
        render_data_chatbot()
    
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
        **Real Estate Market Intelligence Platform**
        
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built with ❤️ using Streamlit & OpenAI*")

if __name__ == "__main__":
    main()