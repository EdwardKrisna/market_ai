import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from openai import OpenAI
import json
import re
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
import requests
import math

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
    .agent-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .selected-agent {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #efe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
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
    
    def get_unique_geographic_values(self, column: str, parent_filter: dict = None, table_name: str = None):
        """Get unique values for geographic columns with optional parent filtering"""
        try:
            if not table_name:
                return []
            
            base_query = f"SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL"
            
            if parent_filter:
                if column == 'wadmkk' and 'wadmpr' in parent_filter:
                    provinces = parent_filter['wadmpr']
                    escaped_provinces = [p.replace("'", "''") for p in provinces]
                    province_list = "', '".join(escaped_provinces)
                    base_query += f" AND wadmpr IN ('{province_list}')"
                elif column == 'wadmkc' and 'wadmkk' in parent_filter:
                    regencies = parent_filter['wadmkk']
                    escaped_regencies = [r.replace("'", "''") for r in regencies]
                    regency_list = "', '".join(escaped_regencies)
                    base_query += f" AND wadmkk IN ('{regency_list}')"
            
            base_query += f" ORDER BY {column}"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(base_query))
                values = [row[0] for row in result.fetchall()]
                return values
        except Exception as e:
            st.error(f"Failed to load {column} options: {str(e)}")
            return []

class CrossAgentQueryParser:
    """Parse cross-agent query syntax"""
    
    def __init__(self):
        self.patterns = {
            'comparison': r'#(\w+)(\s+vs\s+\w+)+',
            'consultation': r'#(\w+)\s+consult\s+([\w\s]+)',
            'market_analysis': r'#all\s+(.+)',
            'impact': r'#(\w+)\s+impact\s+(\w+)'
        }
    
    def parse_query(self, question: str) -> dict:
        """Parse user query for cross-agent triggers"""
        if not question.startswith('#'):
            return {'type': 'single_agent', 'agents': []}
        
        for query_type, pattern in self.patterns.items():
            match = re.match(pattern, question.lower())
            if match:
                return self._extract_agents(query_type, match, question)
        
        return {'type': 'invalid', 'error': 'Invalid cross-agent syntax'}
    
    def _extract_agents(self, query_type: str, match, original_query: str) -> dict:
        """Extract agent names from parsed query"""
        if query_type == 'comparison':
            # Extract all agents from "agent1 vs agent2 vs agent3"
            agents_text = match.group(0)[1:]  # Remove #
            agents = re.findall(r'(\w+)', agents_text)
            # Filter valid agents
            valid_agents = [a for a in agents if a in AGENT_CONFIGS]
            return {
                'type': 'comparison',
                'agents': valid_agents,
                'primary': valid_agents[0] if valid_agents else None,
                'secondary': valid_agents[1:] if len(valid_agents) > 1 else [],
                'original_query': original_query
            }
        elif query_type == 'consultation':
            primary = match.group(1)
            secondary_text = match.group(2)
            secondary = re.findall(r'(\w+)', secondary_text)
            # Filter valid agents
            valid_primary = primary if primary in AGENT_CONFIGS else None
            valid_secondary = [a for a in secondary if a in AGENT_CONFIGS]
            return {
                'type': 'consultation',
                'agents': [valid_primary] + valid_secondary if valid_primary else valid_secondary,
                'primary': valid_primary,
                'secondary': valid_secondary,
                'original_query': original_query
            }
        elif query_type == 'market_analysis':
            return {
                'type': 'market_analysis',
                'agents': list(AGENT_CONFIGS.keys()),
                'primary': 'office',  # Default primary for market analysis
                'secondary': [a for a in AGENT_CONFIGS.keys() if a != 'office'],
                'original_query': original_query
            }
        elif query_type == 'impact':
            agent1 = match.group(1)
            agent2 = match.group(2)
            valid_agents = [a for a in [agent1, agent2] if a in AGENT_CONFIGS]
            return {
                'type': 'impact',
                'agents': valid_agents,
                'primary': valid_agents[1] if len(valid_agents) > 1 else valid_agents[0] if valid_agents else None,
                'secondary': [valid_agents[0]] if len(valid_agents) > 1 else [],
                'original_query': original_query
            }
        
        return {'type': 'invalid', 'error': 'Could not parse query'}

# Add GeocodeService class
class GeocodeService:
    """Handle geocoding using Google Maps Geocoding API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def geocode_address(self, address: str) -> tuple:
        """
        Geocode an address to get latitude and longitude
        Returns: (latitude, longitude, formatted_address) or (None, None, None) if failed
        """
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

class PropertyAIAgent:
    """Individual property AI agent"""
    
    def __init__(self, api_key: str, agent_type: str, db_connection: DatabaseConnection):
        self.client = OpenAI(api_key=api_key)
        self.agent_type = agent_type
        self.config = AGENT_CONFIGS[agent_type]
        self.table_name = self.config['table']
        self.db_connection = db_connection
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create agent-specific system prompt with accurate data types"""
        
        prompts = {
            'retail': """
You are a Retail Property Expert AI for RHR specializing in commercial retail spaces.
Table: retail_converted_2025

RETAIL EXPERTISE:
- Shopping malls, retail outlets, commercial spaces
- Net Lettable Area (NLA) and Gross Floor Area (GFA) analysis
- Retail pricing trends and market analysis
- Developer and project performance
- Grade classification (A, B, C grade retail spaces)

ACCURATE COLUMN DETAILS WITH DATA TYPES:
Basic Information:
- id (INTEGER): Primary key, unique identifier
- geometry (TEXT): PostGIS geometry field
- latitude (DOUBLE PRECISION): Latitude coordinates
- longitude (DOUBLE PRECISION): Longitude coordinates

Project Information:
- project_name (TEXT): Retail development name
- address (TEXT): Property address
- developer (TEXT): Development company
- project_status (TEXT): Development status
- completionyear (INTEGER): Year completed
- q (INTEGER): Quarter of completion

Property Specifications:
- grade (TEXT): Property grade (A, B, C)
- nla (TEXT): Net Lettable Area (rentable space) - stored as TEXT
- gfa (TEXT): Gross Floor Area (total floor space) - stored as TEXT

Pricing Data (ALL TEXT FORMAT):
- price_2016 (TEXT): 2016 pricing data
- price_2017 (TEXT): 2017 pricing data
- price_2018 (TEXT): 2018 pricing data
- price_2019 (TEXT): 2019 pricing data
- price_2020 (TEXT): 2020 pricing data
- price_2021 (TEXT): 2021 pricing data
- price_2022 (TEXT): 2022 pricing data
- price_2023 (TEXT): 2023 pricing data
- price_2024 (TEXT): 2024 pricing data
- price_2025 (TEXT): 2025 pricing data
- price_avg (TEXT): Average price across years

Location Data:
- area (TEXT): General area/district
- precinct (TEXT): Specific precinct
- wadmpr (TEXT): Province
- wadmkk (TEXT): Regency/City  
- wadmkc (TEXT): District

CRITICAL SQL RULES:
1. Price columns are TEXT - use CAST(price_avg AS NUMERIC) for calculations
2. nla and gfa are TEXT - use CAST(nla AS NUMERIC) if numeric operations needed
3. completionyear and q are INTEGER - use directly for numeric operations
4. For price analysis: CAST(price_avg AS NUMERIC) WHERE price_avg IS NOT NULL AND price_avg != ''
5. For map queries: SELECT id, latitude, longitude, project_name, address, grade, price_avg
6. Always handle NULL and empty string values in TEXT columns
7. Use ILIKE for text searches on TEXT columns
8. Geographic filtering: wadmpr, wadmkk, wadmkc are all TEXT
""",
            
            'hospital': """
You are a Hospital Property Expert AI for RHR specializing in healthcare facilities.
Table: hospital_converted_2025

HOSPITAL EXPERTISE:
- Healthcare facility analysis
- Medical capacity and services
- Hospital grades and ownership types
- Healthcare accessibility and coverage
- BPJS and public health services

ACCURATE COLUMN DETAILS WITH DATA TYPES:
Basic Information:
- id (INTEGER): Primary key, unique identifier
- geometry (TEXT): PostGIS geometry field
- latitude (DOUBLE PRECISION): Latitude coordinates
- longitude (DOUBLE PRECISION): Longitude coordinates

Facility Information:
- object_name (TEXT): Hospital/clinic name
- type (TEXT): Healthcare facility type
- grade (TEXT): Hospital grade/classification
- ownership (TEXT): Ownership type (public/private)

Capacity & Infrastructure:
- land_area (TEXT): Land area size - stored as TEXT
- building_area (TEXT): Building area size - stored as TEXT
- beds_capacity (INTEGER): Number of beds - TRUE INTEGER

Services:
- bpjs (TEXT): BPJS coverage (social health insurance)
- kb_gratis (TEXT): Free family planning services

Location Data:
- wadmpr (TEXT): Province
- wadmkk (TEXT): Regency/City
- wadmkc (TEXT): District

CRITICAL SQL RULES:
1. beds_capacity is INTEGER - use directly for SUM(), AVG(), COUNT()
2. land_area and building_area are TEXT - use CAST(land_area AS NUMERIC) for calculations
3. All facility info columns are TEXT - use ILIKE for searches
4. For capacity analysis: SUM(beds_capacity), AVG(beds_capacity) WHERE beds_capacity IS NOT NULL
5. For area analysis: CAST(land_area AS NUMERIC) WHERE land_area IS NOT NULL AND land_area != ''
6. For map queries: SELECT id, latitude, longitude, object_name, type, grade, beds_capacity
7. Use COUNT(*) for facility counts by type, grade, ownership
8. Handle TEXT columns properly with NULL and empty string checks
""",
            
            'office': """
You are an Office Property Expert AI for RHR specializing in commercial office spaces.
Table: office_converted_2025

OFFICE EXPERTISE:
- Commercial office building analysis
- Grade A, B, C office classifications  
- Office rental rates and pricing trends
- Corporate real estate and investment analysis
- Strata Ground Area (SGA) and Gross Floor Area (GFA)

ACCURATE COLUMN DETAILS WITH DATA TYPES:
Basic Information:
- id (INTEGER): Primary key, unique identifier
- geometry (TEXT): PostGIS geometry field
- latitude (DOUBLE PRECISION): Latitude coordinates
- longitude (DOUBLE PRECISION): Longitude coordinates

Building Information:
- building_name (TEXT): Office building name
- grade (TEXT): Building grade (A, B, C)
- project_type (TEXT): Type of office development
- project_status (TEXT): Development status
- completionyear (INTEGER): Year completed
- q (INTEGER): Quarter of completion

Property Specifications:
- sga (TEXT): Strata Ground Area - stored as TEXT
- gfa (TEXT): Gross Floor Area - stored as TEXT
- "owner/developer" (TEXT): Building owner/developer (NOTE: column name has slash)

Pricing Data (TRUE NUMERIC):
- price_2016 (DOUBLE PRECISION): 2016 pricing
- price_2017 (DOUBLE PRECISION): 2017 pricing
- price_2018 (DOUBLE PRECISION): 2018 pricing
- price_2019 (DOUBLE PRECISION): 2019 pricing
- price_2020 (DOUBLE PRECISION): 2020 pricing
- price_2021 (DOUBLE PRECISION): 2021 pricing
- price_2022 (DOUBLE PRECISION): 2022 pricing
- price_2023 (DOUBLE PRECISION): 2023 pricing
- price_2024 (DOUBLE PRECISION): 2024 pricing
- price_2025 (DOUBLE PRECISION): 2025 pricing
- price_avg (DOUBLE PRECISION): Average price

Location Data:
- area (TEXT): General area/district
- precinct (TEXT): Specific precinct
- wadmpr (TEXT): Province
- wadmkk (TEXT): Regency/City
- wadmkc (TEXT): District

CRITICAL SQL RULES:
1. Price columns are DOUBLE PRECISION - use directly: AVG(price_avg), MIN(price_2024), MAX(price_2025)
2. completionyear and q are INTEGER - use directly for numeric operations
3. sga and gfa are TEXT - use CAST(sga AS NUMERIC) for calculations
4. Column name "owner/developer" needs quotes: "owner/developer"
5. For price analysis: SELECT AVG(price_avg) WHERE price_avg IS NOT NULL
6. For map queries: SELECT id, latitude, longitude, building_name, grade, price_avg
7. For area analysis: CAST(gfa AS NUMERIC) WHERE gfa IS NOT NULL AND gfa != ''
8. Use BETWEEN for price ranges: price_avg BETWEEN 1000000 AND 5000000
""",
            
            'condo': """
You are a Condominium Property Expert AI for RHR specializing in residential condominiums.
Table: condo_converted_2025

CONDO EXPERTISE:
- Residential condominium analysis
- Developer performance and project delivery
- Unit counts and residential capacity
- Condo grades and market positioning
- Residential area analysis

ACCURATE COLUMN DETAILS WITH DATA TYPES:
Basic Information:
- id (INTEGER): Primary key, unique identifier
- geometry (TEXT): PostGIS geometry field
- latitude (DOUBLE PRECISION): Latitude coordinates
- longitude (DOUBLE PRECISION): Longitude coordinates

Project Information:
- project_name (TEXT): Condominium project name
- address (TEXT): Property address
- developer (TEXT): Development company
- project_status (TEXT): Development status
- completionyear (INTEGER): Year completed
- q (INTEGER): Quarter of completion

Property Specifications:
- grade (TEXT): Condo grade classification
- unit (INTEGER): Number of units - TRUE INTEGER

Location Data:
- area (TEXT): General area/district
- precinct (TEXT): Specific precinct
- wadmpr (TEXT): Province
- wadmkk (TEXT): Regency/City
- wadmkc (TEXT): District

CRITICAL SQL RULES:
1. unit is INTEGER - use directly: SUM(unit), AVG(unit), COUNT(*)
2. completionyear and q are INTEGER - use directly for numeric operations
3. All text fields are TEXT - use ILIKE for searches
4. NO PRICING DATA available in this table
5. For capacity analysis: SUM(unit) as total_units, AVG(unit) as avg_units_per_project
6. For map queries: SELECT id, latitude, longitude, project_name, address, grade, unit
7. For developer analysis: GROUP BY developer, COUNT(*) as project_count, SUM(unit) as total_units
8. Handle NULL values properly in TEXT columns
9. Use COUNT(*) for project counts by area, grade, status
""",
            
            'hotel': """
You are a Hotel Property Expert AI for RHR specializing in hospitality properties.
Table: hotel_converted_2025

HOTEL EXPERTISE:
- Hotel and hospitality property analysis
- Star rating classifications (1-5 stars)
- Hotel management and operations
- Hospitality pricing and market trends
- Event facilities and capacity analysis

ACCURATE COLUMN DETAILS WITH DATA TYPES:
Basic Information:
- id (INTEGER): Primary key, unique identifier
- geometry (TEXT): PostGIS geometry field
- latitude (DOUBLE PRECISION): Latitude coordinates
- longitude (DOUBLE PRECISION): Longitude coordinates

Hotel Information:
- project_name (TEXT): Hotel name
- address (TEXT): Property address
- developer (TEXT): Development company
- management (TEXT): Hotel management company
- star (TEXT): Star rating
- concept (TEXT): Hotel concept/type
- completionyear (INTEGER): Year completed
- q (INTEGER): Quarter of completion
- project_status (TEXT): Development status

Capacity & Facilities:
- unit_planned (TEXT): Planned units/rooms - stored as TEXT
- unit_developed (INTEGER): Developed units/rooms - TRUE INTEGER
- floors (TEXT): Number of floors - stored as TEXT
- ballroom_capacity (INTEGER): Event space capacity - TRUE INTEGER

Pricing Data (ALL TEXT FORMAT):
- price_2016 (TEXT): 2016 pricing data
- price_2017 (TEXT): 2017 pricing data
- price_2018 (TEXT): 2018 pricing data
- price_2019 (TEXT): 2019 pricing data
- price_2020 (TEXT): 2020 pricing data
- price_2021 (TEXT): 2021 pricing data
- price_2022 (TEXT): 2022 pricing data
- price_2023 (TEXT): 2023 pricing data
- price_2024 (TEXT): 2024 pricing data
- price_2025 (TEXT): 2025 pricing data
- price_avg (TEXT): Average price across years

Location Data:
- area (TEXT): General area/district
- precinct (TEXT): Specific precinct
- wadmpr (TEXT): Province
- wadmkk (TEXT): Regency/City
- wadmkc (TEXT): District

CRITICAL SQL RULES:
1. unit_developed, ballroom_capacity are INTEGER - use directly for calculations
2. completionyear and q are INTEGER - use directly
3. Price columns are TEXT - use CAST(price_avg AS NUMERIC) for calculations
4. unit_planned, floors, and star are TEXT - use CAST(unit_planned AS NUMERIC) if needed
5. For star analysis: AVG(star), COUNT(*) GROUP BY star
6. For capacity analysis: SUM(unit_developed), AVG(ballroom_capacity)
7. For price analysis: CAST(price_avg AS NUMERIC) WHERE price_avg IS NOT NULL AND price_avg != ''
8. For map queries: SELECT id, latitude, longitude, project_name, address, star, concept
9. Handle TEXT columns with NULL and empty string checks
10. Use star rating for quality segmentation: WHERE star >= 4 for luxury hotels
11. For TEXT columns that should be numeric:
- Use: CAST(CASE WHEN column_name ~ '^[0-9]+\.?[0-9]*$' THEN column_name ELSE NULL END AS NUMERIC)
- Filter: WHERE column_name IS NOT NULL AND column_name != '' AND column_name NOT ILIKE '%N/A%'
""",
            'land': """
You are a Land Market Expert AI for RHR specializing in land property analysis and valuation.
Table: engineered_property_data

LAND MARKET EXPERTISE:
- Land valuation and price per meter analysis
- Land characteristics and topographic features
- Geographic market trends and accessibility
- Land development potential assessment
- Area-based pricing comparison

ACCURATE COLUMN DETAILS WITH DATA TYPES:
Basic Information:
- id (INTEGER): Primary key, unique identifier
- alamat (TEXT): Property address
- latitude (TEXT): Latitude coordinates - stored as TEXT
- longitude (TEXT): Longitude coordinates - stored as TEXT
- geometry (GEOMETRY): PostGIS geometry field

Land Specifications:
- luas_tanah (FLOAT): Land area in square meters - TRUE FLOAT
- bentuk_tapak (TEXT): Land shape/configuration
- posisi_tapak (TEXT): Land position/location relative to surroundings
- orientasi (TEXT): Land orientation (North, South, East, West facing)

Environmental & Infrastructure:
- kondisi_wilayah_sekitar (TEXT): Surrounding area conditions
- perkerasan_jalan (TEXT): Road surface condition/type
- jenis_jalan (TEXT): Road type classification

Pricing Data:
- hpm (FLOAT): Harga Per Meter (price per square meter) - TRUE FLOAT
- tahun_pengambilan_data (INTEGER): Data collection year - TRUE INTEGER

Location Data:
- wadmkd (TEXT): Village/Kelurahan
- wadmkc (TEXT): District/Kecamatan
- wadmkk (TEXT): Regency/City
- wadmpr (TEXT): Province

CRITICAL SQL RULES:
1. hpm is FLOAT - use directly: AVG(hpm), MIN(hpm), MAX(hpm) WHERE hpm IS NOT NULL AND hpm > 0
2. luas_tanah is FLOAT - use directly: SUM(luas_tanah), AVG(luas_tanah)
3. latitude/longitude are TEXT - use CAST(latitude AS NUMERIC), CAST(longitude AS NUMERIC)
4. tahun_pengambilan_data is INTEGER - use directly for year filtering
5. For price analysis: AVG(hpm) as avg_price_per_meter, COUNT(*) as sample_count
6. For area analysis: AVG(luas_tanah) as avg_land_area, SUM(luas_tanah) as total_area
7. For map queries: SELECT id, CAST(latitude AS NUMERIC) as lat, CAST(longitude AS NUMERIC) as lng, alamat, hpm, luas_tanah
8. Geographic filtering: wadmpr, wadmkk, wadmkc, wadmkd are all TEXT
9. Handle TEXT coordinates: WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND latitude != '' AND longitude != ''
10. Price filtering: WHERE hpm IS NOT NULL AND hpm > 0 for meaningful price analysis
11. Always include sample counts in aggregations: COUNT(*) as sample_count
"""
        }
        
        base_prompt = f"""
You are a {self.agent_type.title()} Property Expert AI for RHR.
Table: {self.table_name}

You have three helper functions:
  create_map_visualization(sql_query: string, title: string)
    → Returns a map of properties when called.
    
  find_nearby_projects(location_name: string, radius_km: float, title: string)
    → Finds and maps properties near a specific location within given radius.
    
  create_chart_visualization(chart_type: string, sql_query: string, title: string, x_column: string, y_column: string, color_column: string)
    → Creates various charts (bar, pie, line, scatter, histogram) from data.

**RULES**  
- If the user asks for charts/graphs ("grafik", "chart", "barchart", "pie", etc.), use `create_chart_visualization` function.
- If the user asks for properties near a specific location, use `find_nearby_projects` function.
- If the user asks for a general map, use `create_map_visualization` function.  
- Otherwise return *only* a PostgreSQL query (no explanations).

{prompts.get(self.agent_type, '')}

Generate ONLY the PostgreSQL query, no explanations.
"""
        
        return base_prompt
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth (in kilometers)"""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        r = 6371  # Radius of Earth in kilometers
        return c * r

    def create_chart_visualization(self, data: pd.DataFrame, chart_type: str, title: str, 
                                x_col: str = None, y_col: str = None, color_col: str = None) -> str:
        """Create chart visualization using Plotly Express"""
        try:
            if data is None or len(data) == 0:
                return "Error: Tidak ada data untuk membuat grafik."
            
            # Auto-detect columns if not provided
            if x_col is None or y_col is None:
                chart_type, x_col, y_col, color_col = self._suggest_chart_from_dataframe(data, chart_type)
            
            # Ensure columns exist in dataframe
            available_cols = data.columns.tolist()
            if x_col and x_col not in available_cols:
                x_col = available_cols[0] if available_cols else None
            if y_col and y_col not in available_cols:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                y_col = numeric_cols[0] if numeric_cols else available_cols[1] if len(available_cols) > 1 else None
            
            fig = None
            
            # Create chart based on type
            if chart_type == "bar":
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title,
                    labels={x_col: x_col.replace('_', ' ').title(), 
                        y_col: y_col.replace('_', ' ').title() if y_col else 'Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                
            elif chart_type == "pie":
                # For pie charts, use count data or first numeric column
                if y_col:
                    fig = px.pie(
                        data, 
                        names=x_col, 
                        values=y_col, 
                        title=title
                    )
                else:
                    # Count occurrences
                    pie_data = data[x_col].value_counts().reset_index()
                    pie_data.columns = [x_col, 'count']
                    fig = px.pie(
                        pie_data, 
                        names=x_col, 
                        values='count', 
                        title=title
                    )
                    
            elif chart_type == "line":
                fig = px.line(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title,
                    markers=True
                )
                
            elif chart_type == "scatter":
                fig = px.scatter(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title,
                    size_max=60
                )
                
            elif chart_type == "histogram":
                fig = px.histogram(
                    data, 
                    x=x_col if x_col else y_col, 
                    color=color_col,
                    title=title,
                    nbins=20
                )
            
            else:
                # Default to bar chart
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title
                )
                fig.update_layout(xaxis_tickangle=-45)
            
            if fig:
                # Improve chart appearance
                fig.update_layout(
                    height=500,
                    showlegend=True if color_col else False,
                    template="plotly_white",
                    title_x=0.5,
                    margin=dict(l=50, r=50, t=80, b=100)
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                return f"✅ Grafik {chart_type} berhasil ditampilkan dengan {len(data)} data points."
            else:
                return "Error: Gagal membuat grafik."
                
        except Exception as e:
            return f"Error membuat grafik: {str(e)}"

    def _suggest_chart_from_dataframe(self, df: pd.DataFrame, preferred_chart: str = "auto") -> tuple:
        """Suggest best chart configuration from existing dataframe"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove system columns
        system_cols = ['id', 'latitude', 'longitude', 'geometry']
        numeric_cols = [col for col in numeric_cols if col not in system_cols]
        
        x_col, y_col, color_col = None, None, None
        chart_type = preferred_chart
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            if preferred_chart == "auto":
                if 'count' in numeric_cols[0].lower() or len(df) < 50:
                    chart_type = "bar"
                else:
                    chart_type = "scatter"
            
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            color_col = categorical_cols[1] if len(categorical_cols) > 1 else None
            
        elif len(categorical_cols) > 1:
            chart_type = "pie" if preferred_chart == "auto" else preferred_chart
            x_col = categorical_cols[0]
            
        elif len(numeric_cols) > 1:
            chart_type = "scatter" if preferred_chart == "auto" else preferred_chart
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        
        return chart_type, x_col, y_col, color_col

    def find_nearby_properties(self, location_name: str, radius_km: float, title: str, 
                            geocode_service, db_connection) -> str:
        """Find properties near a specific location using geocoding"""
        try:
            if not geocode_service:
                return "Error: Layanan geocoding tidak tersedia. Silakan tambahkan Google Maps API key."
            
            # Geocode the location
            with st.spinner(f"Mencari koordinat untuk '{location_name}'..."):
                lat, lng, formatted_address = geocode_service.geocode_address(location_name)
            
            if lat is None or lng is None:
                return f"Error: Tidak dapat menemukan koordinat untuk lokasi '{location_name}'. Silakan coba dengan nama lokasi yang lebih spesifik."
            
            st.success(f"📍 Lokasi ditemukan: {formatted_address}")
            st.info(f"Koordinat: {lat:.6f}, {lng:.6f}")
            
            # Determine coordinate columns based on agent type
            if self.agent_type == 'land':
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
                {self._get_display_columns()},
                (6371 * acos(
                    cos(radians({lat})) * cos(radians({lat_col})) * 
                    cos(radians({lng_col}) - radians({lng})) + 
                    sin(radians({lat})) * sin(radians({lat_col}))
                )) as distance_km
            FROM {self.table_name}
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
            with st.spinner(f"Mencari properti dalam radius {radius_km} km..."):
                result_df, query_msg = db_connection.execute_query(sql_query)
            
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
                    text_parts.extend(self._create_hover_text(row))
                    text_parts.append(f"Jarak: {row['distance_km']:.2f} km")
                    hover_text.append("<br>".join(text_parts))
                
                # Use agent-specific color
                marker_color = AGENT_CONFIGS[self.agent_type]['color']
                
                fig.add_trace(go.Scattermapbox(
                    lat=result_df['latitude'],
                    lon=result_df['longitude'],
                    mode='markers',
                    marker=dict(size=8, color=marker_color),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name='Properties'
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
                    title=f"{title} - {len(result_df)} properti dalam radius {radius_km} km dari {location_name}"
                )
                
                # Display map
                st.plotly_chart(fig, use_container_width=True)

                # Show results table
                with st.expander("📊 Detail Properti Terdekat", expanded=False):
                    display_cols = ['id'] + [col for col in result_df.columns 
                                        if col not in ['latitude', 'longitude', 'geometry']]
                    st.dataframe(result_df[display_cols].round(2), use_container_width=True)

                return f"✅ Ditemukan {len(result_df)} properti dalam radius {radius_km} km dari {location_name}. Properti terdekat berjarak {result_df['distance_km'].min():.2f} km."
            
            else:
                return f"❌ Tidak ada properti yang ditemukan dalam radius {radius_km} km dari {location_name}."
            
        except Exception as e:
            return f"Error mencari properti terdekat: {str(e)}"

    def _get_display_columns(self) -> str:
        """Get display columns based on agent type"""
        if self.agent_type == 'land':
            return "alamat, hpm, luas_tanah, wadmpr, wadmkk, wadmkc"
        elif self.agent_type == 'condo':
            return "project_name, address, developer, grade, unit, wadmpr, wadmkk"
        elif self.agent_type == 'hotel':
            return "project_name, address, star, management, unit_developed, wadmpr, wadmkk"
        elif self.agent_type == 'office':
            return "building_name, grade, \"owner/developer\", price_avg, wadmpr, wadmkk"
        elif self.agent_type == 'hospital':
            return "object_name, type, grade, beds_capacity, wadmpr, wadmkk"
        elif self.agent_type == 'retail':
            return "project_name, address, developer, grade, price_avg, wadmpr, wadmkk"
        else:
            return "*"

    def _create_hover_text(self, row) -> list:
        """Create hover text based on agent type"""
        if self.agent_type == 'land':
            text_parts = []
            if 'alamat' in row and pd.notna(row['alamat']):
                text_parts.append(f"Alamat: {row['alamat']}")
            if 'hpm' in row and pd.notna(row['hpm']):
                text_parts.append(f"HPM: Rp {row['hpm']:,.0f}/m²")
            if 'luas_tanah' in row and pd.notna(row['luas_tanah']):
                text_parts.append(f"Luas: {row['luas_tanah']:,.0f} m²")
            if 'wadmpr' in row and pd.notna(row['wadmpr']):
                text_parts.append(f"Provinsi: {row['wadmpr']}")
            return text_parts
        else:
            # Default for other agent types
            return [f"Data: {k}={v}" for k, v in row.items() 
                    if k not in ['id', 'latitude', 'longitude', 'geometry', 'distance_km'] 
                    and pd.notna(v)]

    def generate_query(self, user_question: str, geographic_context: str = "") -> dict:
        """Generate SQL query using o4-mini with enhanced tools"""
        
        is_map_request = bool(re.search(r"\b(map|peta|visualisasi lokasi)\b", user_question, re.I))
        is_chart_request = bool(re.search(r"\b(grafik|chart|barchart|pie|line|scatter|histogram|graph|visualisasi data)\b", user_question, re.I))
        is_nearby_request = bool(re.search(r"\b(terdekat|sekitar|dekat|nearby|near)\b", user_question, re.I))
        
        tools = [{
            "type": "function",
            "name": "create_map_visualization",
            "description": "Create a map of properties. Only use when the user explicitly requests location visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": f"SQL query including id, latitude, longitude and relevant columns for {self.table_name}"
                    },
                    "title": { "type": "string" }
                },
                "required": ["sql_query", "title"],
                "additionalProperties": False
            },
            "strict": True
        }, {
            "type": "function",
            "name": "find_nearby_projects",
            "description": "Find and map properties near a specific location. Use when user asks for properties near a place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Name of the location to search near"
                    },
                    "radius_km": {
                        "type": "number",
                        "description": "Search radius in kilometers (default: 1.0)"
                    },
                    "title": { "type": "string" }
                },
                "required": ["location_name", "radius_km", "title"],
                "additionalProperties": False
            },
            "strict": True
        }, {
            "type": "function",
            "name": "create_chart_visualization",
            "description": "Create charts from data. Use when user requests graphs, charts, or data visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "pie", "line", "scatter", "histogram", "auto"],
                        "description": "Type of chart to create"
                    },
                    "sql_query": {
                        "type": "string",
                        "description": "SQL query to get data for the chart"
                    },
                    "title": { "type": "string" },
                    "x_column": {
                        "type": "string",
                        "description": "Column name for x-axis"
                    },
                    "y_column": {
                        "type": "string", 
                        "description": "Column name for y-axis"
                    },
                    "color_column": {
                        "type": "string",
                        "description": "Column name for color grouping"
                    }
                },
                "required": ["chart_type", "sql_query", "title", "x_column", "y_column", "color_column"],
                "additionalProperties": False
            },
            "strict": True
        }]
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        if geographic_context:
            messages.append({"role": "user", "content": geographic_context})
        messages.append({"role": "user", "content": user_question})
        
        # Determine tool choice
        tool_choice = "auto"
        if is_chart_request:
            tool_choice = {"type": "function", "name": "create_chart_visualization"}
        elif is_nearby_request and is_map_request:
            tool_choice = {"type": "function", "name": "find_nearby_projects"}
        elif is_map_request and not is_nearby_request:
            tool_choice = {"type": "function", "name": "create_map_visualization"}
        
        try:
            response = self.client.responses.create(
                model="o4-mini",
                reasoning={"effort": "low"},
                input=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_output_tokens=500
            )
            
            return {'success': True, 'response': response}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def format_response(self, user_question: str, query_results: pd.DataFrame, sql_query: str) -> str:
        """Format response using GPT-4.1-mini"""
        try:
            prompt = f"""User asked: {user_question}

SQL Query executed: {sql_query}
Results: {query_results.to_dict('records') if len(query_results) > 0 else 'No results found'}

Provide clear answer in Bahasa Indonesia. Focus on business insights, not technical details."""

            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                stream=True,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a {self.agent_type.title()} Property Expert. Always respond in Bahasa Indonesia with clear, actionable insights."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            full_response = ""
            response_container = st.empty()
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            return full_response
            
        except Exception as e:
            return f"Maaf, terjadi kesalahan dalam memproses hasil: {str(e)}"
    
    def create_map_visualization(self, query_data: pd.DataFrame, title: str = "Property Locations") -> str:
        """Create map visualization from query data"""
        try:
            if 'latitude' not in query_data.columns or 'longitude' not in query_data.columns:
                return "Error: Data tidak memiliki kolom latitude dan longitude untuk visualisasi peta."
            
            map_df = query_data.copy()
            map_df = map_df.dropna(subset=['latitude', 'longitude'])
            
            map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
            map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
            
            map_df = map_df[
                (map_df['latitude'] >= -90) & (map_df['latitude'] <= 90) &
                (map_df['longitude'] >= -180) & (map_df['longitude'] <= 180)
            ]
            
            if len(map_df) == 0:
                return "Error: Tidak ada data dengan koordinat yang valid untuk visualisasi peta."
            
            fig = go.Figure()
            
            # Create hover text based on available columns
            hover_text = []
            for idx, row in map_df.iterrows():
                text_parts = []
                if 'id' in row:
                    text_parts.append(f"ID: {row['id']}")
                
                # Agent-specific display columns
                if self.agent_type == 'condo' and 'project_name' in row:
                    text_parts.append(f"Project: {row['project_name']}")
                elif self.agent_type == 'hotel' and 'project_name' in row:
                    text_parts.append(f"Hotel: {row['project_name']}")
                elif self.agent_type == 'office' and 'building_name' in row:
                    text_parts.append(f"Building: {row['building_name']}")
                elif self.agent_type == 'hospital' and 'object_name' in row:
                    text_parts.append(f"Hospital: {row['object_name']}")
                elif self.agent_type == 'retail' and 'project_name' in row:
                    text_parts.append(f"Retail: {row['project_name']}")
                
                if 'grade' in row:
                    text_parts.append(f"Grade: {row['grade']}")
                if 'wadmpr' in row:
                    text_parts.append(f"Province: {row['wadmpr']}")
                
                hover_text.append("<br>".join(text_parts))
            
            # Color by agent type
            color = AGENT_CONFIGS[self.agent_type]['color']
            
            fig.add_trace(go.Scattermapbox(
                lat=map_df['latitude'],
                lon=map_df['longitude'],
                mode='markers',
                marker=dict(size=8, color=color),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=f'{self.agent_type.title()} Properties'
            ))
            
            center_lat = map_df['latitude'].mean()
            center_lon = map_df['longitude'].mean()
            
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            return f"✅ Peta berhasil ditampilkan dengan {len(map_df)} properti {self.agent_type}."
            
        except Exception as e:
            return f"Error membuat visualisasi peta: {str(e)}"

def initialize_geocode_service():
        """Initialize geocoding service"""
        try:
            google_api_key = st.secrets["google"]["api_key"]
            if 'geocode_service' not in st.session_state:
                st.session_state.geocode_service = GeocodeService(google_api_key)
            return st.session_state.geocode_service
        except KeyError:
            st.warning("Google Maps API key tidak ditemukan. Fitur pencarian lokasi tidak tersedia.")
            return None

class AgentChatManager:
    """Manage chat histories for all agents"""
    
    def __init__(self):
        self.chat_histories = {agent: [] for agent in AGENT_CONFIGS.keys()}
        self.current_agent = None
    
    def switch_agent(self, new_agent: str):
        """Switch to different agent"""
        self.current_agent = new_agent
        return self.chat_histories[new_agent]
    
    def add_message(self, agent_type: str, role: str, content: str):
        """Add message to agent's chat history"""
        self.chat_histories[agent_type].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_chat_status(self) -> dict:
        """Get overview of all chat sessions"""
        return {
            agent: len(history) 
            for agent, history in self.chat_histories.items()
        }

class MultiAgentManager:
    """Manage multiple property AI agents"""
    
    def __init__(self, api_key: str, db_connection: DatabaseConnection):
        self.api_key = api_key
        self.db_connection = db_connection
        self.agents = {}
        self.parser = CrossAgentQueryParser()
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        for agent_type in AGENT_CONFIGS.keys():
            self.agents[agent_type] = PropertyAIAgent(
                self.api_key, 
                agent_type, 
                self.db_connection
            )
    
    def get_agent(self, agent_type: str) -> PropertyAIAgent:
        """Get specific agent"""
        return self.agents.get(agent_type)
    
    def process_query(self, question: str, current_agent: str, geographic_context: str = "") -> dict:
        """Process query - either single agent or cross-agent"""
        
        # Check if it's a cross-agent query
        parsed = self.parser.parse_query(question)
        
        if parsed['type'] == 'single_agent':
            # Single agent query
            return self._process_single_agent_query(question, current_agent, geographic_context)
        elif parsed['type'] == 'invalid':
            return {
                'type': 'error',
                'message': f"Invalid cross-agent syntax: {parsed.get('error', 'Unknown error')}"
            }
        else:
            # Cross-agent query
            return self._process_cross_agent_query(parsed, geographic_context)
    
    def _handle_function_call(self, output_item, agent, question: str) -> dict:
        """Handle function calls from AI"""
        try:
            args = json.loads(output_item.arguments)
            
            if output_item.name == "create_map_visualization":
                sql_query = args.get("sql_query")
                map_title = args.get("title", "Property Locations")
                
                result_df, query_msg = self.db_connection.execute_query(sql_query)
                
                if result_df is not None and len(result_df) > 0:
                    map_result = agent.create_map_visualization(result_df, map_title)
                    
                    return {
                        'type': 'map',
                        'agent': agent.agent_type,
                        'sql_query': sql_query,
                        'data': result_df,
                        'message': map_result,
                        'title': map_title
                    }
                else:
                    return {
                        'type': 'error',
                        'message': f"Map query failed: {query_msg}"
                    }
            
            elif output_item.name == "find_nearby_projects":
                location_name = args.get("location_name")
                radius_km = args.get("radius_km", 1.0)
                map_title = args.get("title", f"Properties near {location_name}")
                
                # Get geocoding service
                geocode_service = st.session_state.get('geocode_service')
                
                nearby_result = agent.find_nearby_properties(
                    location_name, radius_km, map_title, geocode_service, self.db_connection
                )
                
                return {
                    'type': 'nearby_map',
                    'agent': agent.agent_type,
                    'message': nearby_result,
                    'title': map_title
                }
            
            elif output_item.name == "create_chart_visualization":
                chart_type = args.get("chart_type", "auto")
                sql_query = args.get("sql_query")
                chart_title = args.get("title", "Data Visualization")
                x_col = args.get("x_column")
                y_col = args.get("y_column")
                color_col = args.get("color_column")
                
                result_df, query_msg = self.db_connection.execute_query(sql_query)
                
                if result_df is not None and len(result_df) > 0:
                    chart_result = agent.create_chart_visualization(
                        result_df, chart_type, chart_title, x_col, y_col, color_col
                    )
                    
                    return {
                        'type': 'chart',
                        'agent': agent.agent_type,
                        'sql_query': sql_query,
                        'data': result_df,
                        'message': chart_result,
                        'title': chart_title
                    }
                else:
                    return {
                        'type': 'error',
                        'message': f"Chart query failed: {query_msg}"
                    }
            
            return {
                'type': 'error',
                'message': f"Unknown function: {output_item.name}"
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Function call error: {str(e)}"
            }

    def _process_single_agent_query(self, question: str, agent_type: str, geographic_context: str) -> dict:
        """Process single agent query with enhanced tool support"""
        try:
            agent = self.get_agent(agent_type)
            if not agent:
                return {
                    'type': 'error',
                    'message': f"Agent {agent_type} not found"
                }
            
            # Generate query
            query_result = agent.generate_query(question, geographic_context)
            
            if not query_result['success']:
                return {
                    'type': 'error',
                    'message': f"Query generation failed: {query_result['error']}"
                }
            
            response = query_result['response']
            
            # Check if AI called a function
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'type') and output_item.type == "function_call":
                        return self._handle_function_call(output_item, agent, question)
            
            # Regular SQL query handling
            if hasattr(response, 'output_text'):
                sql_query = response.output_text.strip()
                
                if sql_query and "SELECT" in sql_query.upper():
                    result_df, query_msg = self.db_connection.execute_query(sql_query)
                    
                    if result_df is not None:
                        formatted_response = agent.format_response(question, result_df, sql_query)
                        
                        return {
                            'type': 'query',
                            'agent': agent_type,
                            'sql_query': sql_query,
                            'data': result_df,
                            'message': formatted_response
                        }
                    else:
                        return {
                            'type': 'error',
                            'message': f"Query execution failed: {query_msg}"
                        }
            
            return {
                'type': 'error',
                'message': "No valid response generated"
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Error processing query: {str(e)}"
            }
    
    def _process_cross_agent_query(self, parsed_query: dict, geographic_context: str) -> dict:
        """Process cross-agent query"""
        try:
            agents = parsed_query['agents']
            primary_agent = parsed_query['primary']
            query_type = parsed_query['type']
            original_query = parsed_query['original_query']
            
            if not agents or not primary_agent:
                return {
                    'type': 'error',
                    'message': "Invalid agents specified in cross-agent query"
                }
            
            # Remove the cross-agent syntax from the query
            clean_query = re.sub(r'^#\w+(\s+(vs|consult|impact)\s+\w+)*\s*', '', original_query, flags=re.IGNORECASE)
            
            # Process each agent
            agent_results = {}
            errors = []
            
            for agent_type in agents:
                if agent_type not in AGENT_CONFIGS:
                    errors.append(f"Invalid agent: {agent_type}")
                    continue
                
                try:
                    # Generate agent-specific query
                    agent_query = self._generate_agent_specific_query(clean_query, agent_type, query_type)
                    
                    # Process the query
                    result = self._process_single_agent_query(agent_query, agent_type, geographic_context)
                    
                    if result['type'] == 'error':
                        errors.append(f"{agent_type}: {result['message']}")
                    else:
                        agent_results[agent_type] = result
                        
                except Exception as e:
                    errors.append(f"{agent_type}: {str(e)}")
            
            # Check if we have enough results
            if not agent_results:
                return {
                    'type': 'error',
                    'message': f"All agents failed. Errors: {'; '.join(errors)}"
                }
            
            # Combine results
            combined_result = self._combine_cross_agent_results(
                agent_results, 
                primary_agent, 
                query_type, 
                clean_query,
                errors
            )
            
            return combined_result
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Cross-agent query failed: {str(e)}"
            }
    
    def _generate_agent_specific_query(self, query: str, agent_type: str, query_type: str) -> str:
        """Generate agent-specific query based on cross-agent type"""
        
        if query_type == 'comparison':
            # For comparison, ask each agent about their domain
            domain_map = {
                'condo': 'residential condominium',
                'hotel': 'hotel',
                'office': 'office building',
                'hospital': 'hospital',
                'retail': 'retail space'
            }
            domain = domain_map.get(agent_type, agent_type)
            return f"Analyze {domain} data for: {query}"
        
        elif query_type == 'consultation':
            # For consultation, the primary agent asks about the query
            return f"Provide data analysis for: {query}"
        
        elif query_type == 'market_analysis':
            # For market analysis, each agent analyzes their market
            return f"Provide market analysis for {agent_type} properties: {query}"
        
        elif query_type == 'impact':
            # For impact analysis, analyze the relationship
            return f"Analyze impact relationship: {query}"
        
        return query
    
    def _combine_cross_agent_results(self, agent_results: dict, primary_agent: str, 
                                   query_type: str, query: str, errors: list) -> dict:
        """Combine results from multiple agents"""
        
        try:
            # Get primary agent for final formatting
            primary_agent_obj = self.get_agent(primary_agent)
            
            # Combine all data
            combined_data = pd.DataFrame()
            agent_summaries = {}
            
            for agent_type, result in agent_results.items():
                if result['type'] == 'query' and 'data' in result:
                    # Add agent identifier to data
                    agent_data = result['data'].copy()
                    agent_data['source_agent'] = agent_type
                    
                    # Combine data
                    if combined_data.empty:
                        combined_data = agent_data
                    else:
                        # For cross-agent queries, we might want to keep data separate
                        combined_data = pd.concat([combined_data, agent_data], ignore_index=True)
                    
                    # Store summary
                    agent_summaries[agent_type] = {
                        'message': result['message'],
                        'sql_query': result['sql_query'],
                        'row_count': len(result['data'])
                    }
            
            # Format combined response
            combined_message = self._format_cross_agent_response(
                agent_summaries, 
                primary_agent, 
                query_type, 
                query, 
                primary_agent_obj,
                errors
            )
            
            return {
                'type': 'cross_agent',
                'primary_agent': primary_agent,
                'query_type': query_type,
                'agents': list(agent_results.keys()),
                'data': combined_data,
                'message': combined_message,
                'agent_summaries': agent_summaries,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Failed to combine cross-agent results: {str(e)}"
            }
    
    def _format_cross_agent_response(self, agent_summaries: dict, primary_agent: str, 
                                   query_type: str, query: str, primary_agent_obj: PropertyAIAgent,
                                   errors: list) -> str:
        """Format cross-agent response using primary agent"""
        
        try:
            # Create summary of all agent results
            summary_text = f"Cross-agent {query_type} analysis for: {query}\n\n"
            
            # Add results from each agent
            for agent_type, summary in agent_summaries.items():
                agent_name = AGENT_CONFIGS[agent_type]['name']
                summary_text += f"=== {agent_name} Analysis ===\n"
                summary_text += f"Data points: {summary['row_count']}\n"
                summary_text += f"Analysis: {summary['message']}\n\n"
            
            # Add errors if any
            if errors:
                summary_text += f"=== Warnings ===\n"
                for error in errors:
                    summary_text += f"- {error}\n"
                summary_text += "\n"
            
            # Use primary agent to format the final response
            prompt = f"""
Multiple property experts have analyzed the query: {query}

Query Type: {query_type}

Results from different experts:
{summary_text}

Please provide a comprehensive analysis in Bahasa Indonesia that:
1. Synthesizes insights from all experts
2. Highlights key findings and patterns
3. Provides actionable recommendations
4. Compares different property types where relevant

Focus on business insights and strategic recommendations.
"""
            
            response = primary_agent_obj.client.chat.completions.create(
                model="gpt-4.1-mini",
                stream=True,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are the lead {primary_agent.title()} Property Expert coordinating a multi-agent analysis. Provide comprehensive insights in Bahasa Indonesia."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.3
            )
            
            full_response = ""
            response_container = st.empty()
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            
            return full_response
            
        except Exception as e:
            return f"Error formatting cross-agent response: {str(e)}"


def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def login():
    """Handle user login"""
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

def initialize_database():
    """Initialize database connection"""
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

def initialize_session_state():
    """Initialize session state with geocoding service"""
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = AgentChatManager()
    
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = 'condo'
    
    if 'geographic_filters' not in st.session_state:
        st.session_state.geographic_filters = {}
    
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = None
    
    # Initialize geocoding service
    if 'geocode_service' not in st.session_state:
        st.session_state.geocode_service = initialize_geocode_service()

def render_agent_selection():
    """Render agent selection interface"""
    st.markdown('<div class="section-header">🤖 Select Property Expert</div>', unsafe_allow_html=True)
    
    # Agent selection cards
    cols = st.columns(3)
    
    for i, (agent_type, config) in enumerate(AGENT_CONFIGS.items()):
        with cols[i % 3]:
            # Check if this agent is selected
            is_selected = st.session_state.current_agent == agent_type
            
            # Get chat status
            chat_status = st.session_state.chat_manager.get_chat_status()
            message_count = chat_status.get(agent_type, 0)
            status_indicator = "💬" if message_count > 0 else "💤"
            
            # Create the card with button
            if st.button(
                f"{config['icon']} {config['name']}\n{config['description']}\n{status_indicator} {message_count} messages", 
                key=f"agent_{agent_type}",
                use_container_width=True
            ):
                st.session_state.current_agent = agent_type
                st.session_state.chat_manager.switch_agent(agent_type)
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

def render_geographic_filter():
    """Render geographic filtering interface"""
    st.markdown('<div class="section-header">🌍 Geographic Filter (Optional)</div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    # Get the current agent's table for geographic filtering
    current_table = AGENT_CONFIGS[st.session_state.current_agent]['table']
    
    st.markdown("Select geographic areas to focus analysis (optional)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Province**")
        if st.button("Load Provinces", key="load_provinces"):
            with st.spinner("Loading provinces..."):
                province_options = st.session_state.db_connection.get_unique_geographic_values(
                    'wadmpr', table_name=current_table
                )
                st.session_state.province_options = province_options
        
        if 'province_options' in st.session_state:
            selected_provinces = st.multiselect(
                "Select Provinces",
                st.session_state.province_options,
                key="selected_provinces"
            )
        else:
            selected_provinces = []
            st.info("Click 'Load Provinces' to see options")
    
    with col2:
        st.markdown("**Regency/City**")
        if selected_provinces and st.button("Load Regencies", key="load_regencies"):
            with st.spinner("Loading regencies..."):
                regency_options = st.session_state.db_connection.get_unique_geographic_values(
                    'wadmkk',
                    {'wadmpr': selected_provinces},
                    table_name=current_table
                )
                st.session_state.regency_options = regency_options
        
        if 'regency_options' in st.session_state and selected_provinces:
            selected_regencies = st.multiselect(
                "Select Regencies",
                st.session_state.regency_options,
                key="selected_regencies"
            )
        else:
            selected_regencies = []
            if not selected_provinces:
                st.info("Select provinces first")
            else:
                st.info("Click 'Load Regencies' to see options")
    
    with col3:
        st.markdown("**District**")
        if selected_regencies and st.button("Load Districts", key="load_districts"):
            with st.spinner("Loading districts..."):
                district_options = st.session_state.db_connection.get_unique_geographic_values(
                    'wadmkc',
                    {'wadmkk': selected_regencies},
                    table_name=current_table
                )
                st.session_state.district_options = district_options
        
        if 'district_options' in st.session_state and selected_regencies:
            selected_districts = st.multiselect(
                "Select Districts",
                st.session_state.district_options,
                key="selected_districts"
            )
        else:
            selected_districts = []
            if not selected_regencies:
                st.info("Select regencies first")
            else:
                st.info("Click 'Load Districts' to see options")
    
    # Store geographic filters
    st.session_state.geographic_filters = {
        'wadmpr': selected_provinces,
        'wadmkk': selected_regencies,
        'wadmkc': selected_districts
    }
    
    # Show current selection
    if any([selected_provinces, selected_regencies, selected_districts]):
        st.markdown("**Current Selection:**")
        if selected_provinces:
            st.write(f"🌏 Provinces: {', '.join(selected_provinces)}")
        if selected_regencies:
            st.write(f"🏙️ Regencies: {', '.join(selected_regencies)}")
        if selected_districts:
            st.write(f"📍 Districts: {', '.join(selected_districts)}")
        
        if st.button("Clear Filters"):
            for key in ['province_options', 'regency_options', 'district_options']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.geographic_filters = {}
            st.rerun()
    else:
        st.info("No geographic filters applied")

def render_ai_chat():
    """Render AI chat interface with enhanced tool support"""
    st.markdown('<div class="section-header">💬 AI Chat</div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    # Initialize geocoding service
    geocode_service = initialize_geocode_service()
    
    # Get API key
    try:
        api_key = st.secrets["openai"]["api_key"]
    except KeyError:
        st.error("OpenAI API key not found in secrets.toml")
        return
    
    # Initialize agent manager with geocoding service
    if st.session_state.agent_manager is None:
        st.session_state.agent_manager = MultiAgentManager(api_key, st.session_state.db_connection)
        # Store geocoding service in session state
        st.session_state.geocode_service = geocode_service
    
    # Get current agent's chat history
    current_history = st.session_state.chat_manager.switch_agent(st.session_state.current_agent)
    
    # Display geocoding service status
    if geocode_service:
        st.success("🌍 Layanan pencarian lokasi aktif - dapat mencari properti terdekat")
    else:
        st.warning("⚠️ Layanan pencarian lokasi tidak aktif - tambahkan Google Maps API key untuk fitur pencarian terdekat")
    
    # Display geographic context
    if any(st.session_state.geographic_filters.values()):
        st.markdown("**Geographic Context:**")
        context_parts = []
        for key, values in st.session_state.geographic_filters.items():
            if values:
                context_parts.append(f"{key}: {', '.join(values)}")
        st.info(" | ".join(context_parts))
    
    # Display chat history
    for message in current_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about properties..."):
        # Add user message
        st.session_state.chat_manager.add_message(
            st.session_state.current_agent, 
            "user", 
            prompt
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            # Build geographic context
            geo_context = ""
            if any(st.session_state.geographic_filters.values()):
                context_parts = []
                for key, values in st.session_state.geographic_filters.items():
                    if values:
                        context_parts.append(f"{key}: {values}")
                geo_context = "Geographic context: " + " | ".join(context_parts)
            
            # Show cross-agent indicator if detected
            if prompt.startswith('#'):
                st.markdown('<div class="cross-agent-indicator">🔗 Cross-Agent Query Detected</div>', unsafe_allow_html=True)
            
            # Process the query
            result = st.session_state.agent_manager.process_query(
                prompt, 
                st.session_state.current_agent, 
                geo_context
            )
            
            if result['type'] == 'error':
                st.error(result['message'])
                response_message = f"Error: {result['message']}"
            
            elif result['type'] == 'query':
                # Show query results in expandable section
                with st.expander("📊 Query Details", expanded=False):
                    st.code(result['sql_query'], language="sql")
                    if 'data' in result and not result['data'].empty:
                        st.dataframe(result['data'], use_container_width=True)
                
                response_message = result['message']
            
            elif result['type'] == 'map':
                # Map was already displayed by the agent
                with st.expander("📊 Map Query Details", expanded=False):
                    st.code(result['sql_query'], language="sql")
                    if 'data' in result and not result['data'].empty:
                        st.dataframe(result['data'], use_container_width=True)
                
                response_message = result['message']
            
            elif result['type'] == 'nearby_map':
                # Nearby map was already displayed
                response_message = result['message']
            
            elif result['type'] == 'chart':
                # Chart was already displayed by the agent
                with st.expander("📊 Chart Query Details", expanded=False):
                    st.code(result['sql_query'], language="sql")
                    if 'data' in result and not result['data'].empty:
                        st.dataframe(result['data'], use_container_width=True)
                
                response_message = result['message']
            
            elif result['type'] == 'cross_agent':
                # Show cross-agent results
                st.markdown(f"**Cross-Agent {result['query_type'].title()} Analysis**")
                st.markdown(f"Primary Agent: {result['primary_agent'].title()}")
                st.markdown(f"Participating Agents: {', '.join([a.title() for a in result['agents']])}")
                
                # Show individual agent results in expander
                with st.expander("📊 Individual Agent Results", expanded=False):
                    for agent_type, summary in result['agent_summaries'].items():
                        st.markdown(f"**{AGENT_CONFIGS[agent_type]['name']}**")
                        st.code(summary['sql_query'], language="sql")
                        st.write(f"Rows: {summary['row_count']}")
                        st.markdown("---")
                
                # Show warnings if any
                if result['errors']:
                    with st.expander("⚠️ Warnings", expanded=False):
                        for error in result['errors']:
                            st.warning(error)
                
                response_message = result['message']
            
            else:
                response_message = "Unknown response type"
            
            # Add assistant response to history
            st.session_state.chat_manager.add_message(
                st.session_state.current_agent, 
                "assistant", 
                response_message
            )
    
    # Chat management
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_manager.chat_histories[st.session_state.current_agent] = []
            st.rerun()
    
    with col2:
        if st.button("Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "agent": st.session_state.current_agent,
                "geographic_filters": st.session_state.geographic_filters,
                "chat_history": st.session_state.chat_manager.chat_histories[st.session_state.current_agent]
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
        "🌍 Geographic Filter", 
        "💬 AI Chat"
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
    elif page == "🌍 Geographic Filter":
        render_geographic_filter()
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
    
    # Current agent
    if st.session_state.current_agent:
        config = AGENT_CONFIGS[st.session_state.current_agent]
        st.sidebar.success(f"🤖 Agent: {config['icon']} {config['name']}")
    
    # Geographic filters
    if any(st.session_state.geographic_filters.values()):
        filter_count = sum(len(v) for v in st.session_state.geographic_filters.values() if v)
        st.sidebar.success(f"🌍 Filters: {filter_count} applied")
    else:
        st.sidebar.info("🌍 No geographic filters")
    
    # Chat status
    chat_status = st.session_state.chat_manager.get_chat_status()
    active_chats = sum(1 for count in chat_status.values() if count > 0)
    total_messages = sum(chat_status.values())
    st.sidebar.info(f"💬 Chats: {active_chats} active, {total_messages} total messages")
    
    # Agent status breakdown
    st.sidebar.markdown("**🤖 Agent Status**")
    for agent_type, count in chat_status.items():
        config = AGENT_CONFIGS[agent_type]
        status = "💬" if count > 0 else "💤"
        st.sidebar.text(f"{status} {config['icon']} {config['name']}: {count}")

if __name__ == "__main__":
    main()