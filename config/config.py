"""
Configuration management for Halo Effect Analysis
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Google Cloud Platform Configuration
GCP_KEY_PATH = os.getenv('GCP_KEY_PATH', str(PROJECT_ROOT / 'config' / 'gcp_key.json'))
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-project-id')

# Data Directories
DATA_DIR = Path(os.getenv('DATA_DIR', str(PROJECT_ROOT / 'data')))
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output Directories
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(PROJECT_ROOT / 'output')))
CLUSTERS_OUTPUT_DIR = OUTPUT_DIR / 'clusters'
PARAMS_OUTPUT_DIR = OUTPUT_DIR / 'params'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CLUSTERS_OUTPUT_DIR, PARAMS_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Processing Configuration
NUM_WORKERS = int(os.getenv('NUM_WORKERS', 50))  # Multiprocessing pool size

# BigQuery Configuration
BIGQUERY_ALLOW_LARGE_RESULTS = True

# Data File Paths - BABY
BABY_TRANSACTION_DATA = RAW_DATA_DIR / 'BABY_Instore_Transactions_v2.pkl'
BABY_DISTANCE_DATA = RAW_DATA_DIR / 'BABY_Distance_data_full.pkl'
BABY_HOME_STORE_DATA = RAW_DATA_DIR / 'BABY_Home_Store_full.pkl'
BABY_BOPIS_DATA = RAW_DATA_DIR / 'BABY_BOPIS_v2.pkl'

# Data File Paths - BBBY Open Stores
BBBY_TRANSACTION_DATA = RAW_DATA_DIR / 'BBBY_Open_Store_Transactions_v2.pkl'
BBBY_DISTANCE_DATA = RAW_DATA_DIR / 'BBBY_Open_Stores_Distance_data_full.pkl'
BBBY_BOPIS_DATA = RAW_DATA_DIR / 'BBBY_open_store_BOPIS_v2.pkl'
BBBY_BEST_PARAMS = RAW_DATA_DIR / 'BBBY_best_params_stores.csv'

# Data File Paths - BBBY Closed Stores
BBBY_CLOSED_TRANSACTION_DATA = RAW_DATA_DIR / 'BBBY_closed_Instore_Transactions_v2.pkl'
BBBY_CLOSED_DISTANCE_DATA = RAW_DATA_DIR / 'BBBY_closed_store_Distance_data_full.pkl'
BBBY_CLOSED_BOPIS_DATA = RAW_DATA_DIR / 'BBBY_closed_store_BOPIS_v2.pkl'
CLOSURE_BEST_PARAMS = RAW_DATA_DIR / 'closure_best_params_stores.csv'

# Data File Paths - Forecasting
OPEN_STORES_DATA = RAW_DATA_DIR / 'open_Stores_phase1_17_18_19_20_21.csv'
CLOSED_STORES_DATA = RAW_DATA_DIR / 'closed_Stores_phase1_17_18_19_20_21.csv'
STORE_CLUSTERS_DATA = RAW_DATA_DIR / 'Store_Clusters.xlsx'

# DBSCAN Parameters
DBSCAN_EPS_RANGE = (0.1, 0.9, 0.1)  # start, stop, step
DBSCAN_MIN_SAMPLES_RANGE = (1, 10, 1)  # start, stop, step

# Distance Parameters
MAX_DISTANCE = 50  # Maximum distance for trade area consideration (miles)

# Output File Naming
BABY_CLUSTERS_PREFIX = 'baby_'
BBBY_CLUSTERS_PREFIX = 'bbby_'
CLOSED_CLUSTERS_PREFIX = 'closed_'

# BigQuery Queries
CENSUS_QUERY = """
SELECT distinct geo_id, pop_25_years_over
FROM `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`
"""

BABY_OPEN_STORES_QUERY = """
SELECT distinct BABY_OPEN_STORES
FROM `dw-bq-data-d00.QUANT_STG.SK_BABY_OPEN_STORES`
"""

BBBY_OPEN_STORES_QUERY = """
SELECT distinct BBBY__OPEN_STORES
FROM `dw-bq-data-d00.QUANT_STG.SK_OPEN_STORES_JULY_2021`
"""

BBBY_CLOSED_STORES_QUERY = """
SELECT distinct Store__
FROM `dw-bq-data-d00.QUANT_STG.sk_store_closings`
WHERE Concept = 'BBBY'
  AND extract(year from Close_Date) = 2020
  AND extract(month from Close_Date) = 12
"""

# Prophet Forecasting Configuration
PROPHET_SEASONALITY_MODE = 'multiplicative'
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.001
FORECAST_PERIODS = 18  # Number of months to forecast
FORECAST_FREQ = 'MS'  # Month start frequency

def validate_config():
    """
    Validate that all required configuration is present
    """
    issues = []

    # Check GCP key path
    if not os.path.exists(GCP_KEY_PATH):
        issues.append(f"GCP key file not found at: {GCP_KEY_PATH}")

    # Check if project ID is still default
    if GCP_PROJECT_ID == 'your-project-id':
        issues.append("GCP_PROJECT_ID not set. Please update .env file")

    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True

def print_config():
    """
    Print current configuration (excluding sensitive data)
    """
    print("=" * 60)
    print("Halo Effect Analysis - Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"GCP Project ID: {GCP_PROJECT_ID}")
    print(f"GCP Key Present: {os.path.exists(GCP_KEY_PATH)}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
    if validate_config():
        print("\nConfiguration is valid!")
    else:
        print("\nPlease fix configuration issues before running.")
