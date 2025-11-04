# Halo Effect Analysis

A comprehensive data science framework for quantifying the "halo effect" of retail store closures on nearby locations using advanced machine learning clustering, geospatial analysis, and time series forecasting.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Business Problem](#business-problem)
- [What is the Halo Effect?](#what-is-the-halo-effect)
- [Project Overview](#project-overview)
- [Technical Architecture](#technical-architecture)
- [Methodology Deep Dive](#methodology-deep-dive)
- [Data Sources & Requirements](#data-sources--requirements)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Output & Results](#output--results)
- [Performance & Scalability](#performance--scalability)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [References](#references)

---

## Executive Summary

When a retail store closes, its former customers must shop elsewhere. This redistribution of customers to nearby competing stores creates a "halo effect" - a measurable increase in sales at surrounding locations. This project provides a sophisticated analytical framework to:

1. **Identify trade areas** for each store using census-level demographic and transaction data
2. **Predict customer redistribution** patterns when stores close
3. **Forecast sales impact** on nearby stores, controlling for external factors like COVID-19
4. **Build predictive models** to estimate closure impact magnitude

**Real-world applications:**
- Strategic planning for store closures or openings
- Competitive intelligence and market analysis
- Real estate investment decisions
- Supply chain optimization
- COVID-19 pandemic impact assessment

---

## Business Problem

### The Challenge

Retail companies face critical strategic questions:

1. **When closing underperforming stores:**
   - Which nearby stores will benefit from customer redistribution?
   - How much additional revenue can we expect at nearby locations?
   - Will the increased revenue offset the loss from closure?
   - How far will customers travel to alternate locations?

2. **When evaluating new store locations:**
   - What is the true trade area for a potential site?
   - How will it impact existing stores (cannibalization)?
   - What market share can we capture from competitors?

3. **During external disruptions (e.g., COVID-19):**
   - How do we separate pandemic impact from closure impact?
   - Which stores are most resilient or vulnerable?
   - How have customer shopping patterns changed?

### Traditional Approaches Fall Short

**Problem:** Simple radius-based trade areas (e.g., "all customers within 5 miles") are inaccurate because:
- Population density varies dramatically by location
- Customer behavior is multi-dimensional (distance, transactions, online pickup)
- Geographic barriers (highways, rivers) affect accessibility
- Demographic factors influence shopping patterns
- Competitive landscape matters

**This project's solution:** Machine learning clustering that considers multiple dimensions simultaneously to identify true trade areas with precision.

---

## What is the Halo Effect?

### Definition

The **halo effect** in retail is the phenomenon where closing one store location results in increased sales at nearby stores as customers are redistributed. It's called a "halo" because the impact radiates outward from the closed location to surrounding stores.

### Key Concepts

1. **Trade Area**: The geographic region from which a store draws its customers
   - Primary trade area: 60-70% of customers
   - Secondary trade area: 20-30% of customers
   - Tertiary trade area: Remaining customers

2. **Customer Redistribution**: When a store closes:
   - Former customers choose alternative locations
   - Some go to the next-closest store
   - Some switch to competitors
   - Some shift to online shopping
   - Pattern depends on distance, store similarity, product availability

3. **Closure Factor**: A quantitative measure of impact:
   - Closure Factor = (Post-closure sales - Pre-closure baseline) / Pre-closure sales
   - Example: Closure Factor of 0.15 = 15% sales increase at nearby store
   - Varies by store based on proximity, capacity, and customer overlap

### Real-World Example

**Scenario:** A Bed Bath & Beyond store closes in a suburban shopping center

**Without Halo Effect Analysis:**
- Company loses 100% of that store's revenue
- Uncertainty about customer behavior
- Potential missed opportunity to prepare nearby stores

**With Halo Effect Analysis:**
- Identify that 3 nearby stores are within customers' trade areas
- Predict 12% sales increase at the closest store (3 miles away)
- Predict 7% increase at another store (5 miles away)
- Total recovery: ~45% of lost revenue captured at remaining stores
- Actionable insights: Increase inventory at benefiting stores, adjust staffing

---

## Project Overview

### What This Project Does

This framework performs four major analytical tasks:

#### 1. Trade Area Identification (BABY & BBBY Stores)

**Purpose:** Define precise geographic boundaries for each store's customer base

**How it works:**
- Analyzes transaction data at census block level (~1,000 people per block)
- Combines multiple signals:
  - In-store transaction density (normalized by population)
  - Customer-to-store distance
  - BOPIS (Buy Online Pickup In Store) activity
- Uses DBSCAN clustering algorithm to identify natural boundaries
- Optimizes cluster parameters automatically using silhouette scores
- Two-phase approach for refinement of ambiguous areas

**Output:** Census block-level assignments showing which stores serve which neighborhoods

#### 2. Closed Store Trade Area Analysis

**Purpose:** Understand trade areas for stores that have closed

**How it works:**
- Uses pre-computed optimal clustering parameters
- Applies single-phase DBSCAN with KMeans refinement
- Identifies primary vs. secondary trade areas
- Maps former customers to potential receiving stores

**Output:** Historical trade area definitions for closed locations

#### 3. COVID-19 Impact Forecasting

**Purpose:** Separate pandemic effects from closure effects using time series analysis

**How it works:**
- Uses Prophet (Facebook's forecasting library) for time series prediction
- Trains on pre-pandemic data (2017-2019)
- Forecasts expected performance without COVID-19 (2020-2021)
- Compares actual vs. predicted to isolate pandemic impact
- Analyzes both open stores (to quantify COVID effect) and closed stores (to quantify closure effect)

**Output:** Store-level COVID impact factors and closure impact predictions

#### 4. Regression Modeling (Jupyter Notebook)

**Purpose:** Build predictive models for closure factor magnitude

**How it works:**
- Combines trade area data with transaction history
- Uses store attributes, demographic data, and distance metrics as features
- Builds regression models to predict expected halo effect
- Validates against actual post-closure data

**Output:** Predictive model for estimating closure impact on specific stores

### Brands Analyzed

- **BBBY (Bed Bath & Beyond)**: Home goods retailer with ~700+ locations
- **BABY (Buy Buy Baby)**: Baby products retailer (BBBY subsidiary) with ~100+ locations

Both brands share similar:
- Customer demographics (primarily suburban families)
- Shopping patterns (mix of in-store and online with pickup)
- Store formats (big-box retail)
- Geographic distribution (primarily US markets)

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
├─────────────────────────────────────────────────────────────┤
│  • Google BigQuery (Census data)                            │
│  • Transaction databases (Pickle files)                     │
│  • Geolocation data (Census block coordinates)              │
│  • Historical sales data (CSV)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  1. Data Loading & Validation                               │
│  2. Feature Engineering                                      │
│     - FIPS code standardization                             │
│     - Population normalization                              │
│     - Distance transformation                               │
│  3. Feature Scaling (StandardScaler)                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           Machine Learning & Analytics                       │
├─────────────────────────────────────────────────────────────┤
│  • DBSCAN Clustering (Trade Areas)                          │
│  • KMeans Refinement (Primary/Secondary Areas)              │
│  • Prophet Time Series (COVID Forecasting)                  │
│  • Regression Models (Closure Factor Prediction)            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                Output & Results                              │
├─────────────────────────────────────────────────────────────┤
│  • Cluster assignments (CSV per store)                      │
│  • Model parameters (CSV per store)                         │
│  • Forecast predictions (Aggregate CSV)                     │
│  • Closure factor estimates (Model outputs)                 │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Halo-Effect/
├── src/
│   └── halo_effect/                    # Main package
│       ├── __init__.py                 # Package initialization
│       ├── baby_trade_area.py          # BABY stores analysis
│       ├── bbby_trade_area.py          # BBBY open stores analysis
│       ├── closed_stores_trade_area.py # Closed stores analysis
│       └── forecasting_covid_halo.py   # COVID-19 forecasting
│
├── config/
│   ├── config.py                       # Configuration management
│   └── .env.example                    # Environment template
│
├── data/
│   ├── raw/                            # Input data files
│   │   ├── *_Transactions_v2.pkl       # Transaction history
│   │   ├── *_Distance_data_full.pkl    # Geographic distances
│   │   ├── *_BOPIS_v2.pkl              # Online pickup data
│   │   └── *.csv                       # Parameters and sales data
│   └── processed/                      # Intermediate outputs
│
├── output/
│   ├── clusters/                       # Trade area assignments
│   │   └── {brand}_{store}_clusters.csv
│   └── params/                         # Model parameters
│       └── {brand}_{store}_params.csv
│
├── notebooks/
│   ├── Closure_Factor_Regression_Model_BBBY.ipynb
│   └── Regression_Model_Data.sql       # BigQuery queries
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installer
└── README.md                           # This file
```

---

## Methodology Deep Dive

### 1. DBSCAN Clustering for Trade Areas

#### Why DBSCAN?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is ideal for this problem because:

1. **No predefined cluster count**: Unlike K-means, DBSCAN discovers the natural number of trade areas
2. **Handles irregular shapes**: Real trade areas don't form perfect circles
3. **Identifies outliers**: Can flag unusual customer locations (noise points)
4. **Density-based**: Matches the reality that customers cluster in high-density residential areas

#### Algorithm Steps

**Phase 1: Initial Clustering**

```python
# Input features (all standardized):
X = [
    instore_transaction_density,  # Transactions per capita
    customer_store_distance,      # Miles (inverted: 50 - distance)
    bopis_order_density          # Online pickups per capita
]

# Parameter optimization:
for eps in [0.1, 0.2, ..., 0.9]:
    for min_samples in [1, 2, ..., 9]:
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        score = silhouette_score(X, clusters.labels_)

# Select parameters with highest silhouette score
best_params = argmax(scores)
```

**Phase 2: Refinement**

Cluster 0 (noise/outliers) often contains mixed-boundary areas. The algorithm:
1. Extracts all points labeled as cluster 0
2. Runs a second DBSCAN optimization on these points
3. Re-assigns points that form coherent sub-clusters
4. Keeps true outliers as noise

#### Feature Engineering Details

**1. Transaction Density Normalization**

Raw transaction counts are misleading:
- Census block A: 100 transactions, 500 residents = 0.20 transactions/person
- Census block B: 50 transactions, 100 residents = 0.50 transactions/person

Block B has stronger affinity despite fewer absolute transactions.

```python
# Merge transaction data with census population
transaction_density = transaction_count / population_25_years_over
```

**2. Distance Transformation**

Customers prefer closer stores, so distance has an inverse relationship:

```python
# Transform so higher values = closer stores
distance_transformed = MAX_DISTANCE - actual_distance  # 50 - distance
```

**3. BOPIS Integration**

Buy Online Pickup In Store behavior indicates strong store affinity:
- Customers willing to drive to specific store
- Planned shopping trips (not impulse)
- Shows store preference over convenience

```python
bopis_density = bopis_customer_count / population_25_years_over
```

#### Output Interpretation

Each store gets a CSV file with census blocks assigned to its trade area:

```csv
FIPS_STATE_CD,FIPS_COUNTY_CD,CENSUS_BLOCK_CD,GEOID,Store Number
36,061,040200,36061040200,2001
36,061,040300,36061040300,2001
36,061,040400,36061040400,2001
```

This means:
- Census blocks 36061040200, 36061040300, 36061040400 are in Store 2001's trade area
- These blocks are in New York (36), Manhattan (061)
- Can be mapped geographically for visualization

---

### 2. Prophet Time Series Forecasting

#### Purpose

Separate COVID-19 impact from store closure impact by forecasting counterfactual scenarios.

#### Why Prophet?

Prophet (developed by Facebook) excels at retail forecasting because:

1. **Handles multiple seasonalities**: Daily, weekly, yearly patterns in retail
2. **Robust to missing data**: Works even with gaps in time series
3. **Automatic changepoint detection**: Identifies trend shifts
4. **Multiplicative seasonality**: Better for retail (sales scale with baseline)
5. **Interpretable**: Decomposes forecast into trend + seasonality + holidays

#### Training Approach

**Data Split:**
- Training: 2017-2019 (pre-pandemic normal operations)
- Test: 2020-2021 (pandemic period)

**Model Configuration:**
```python
model = Prophet(
    seasonality_mode='multiplicative',  # Retail has scaling seasonality
    changepoint_prior_scale=0.001       # Conservative (avoid overfitting)
)
```

**Forecast Process:**
1. Train on pre-COVID historical data
2. Predict through 2020-2021 (18 months)
3. Compare predicted vs. actual sales
4. Calculate percentage difference = COVID impact

#### Comparative Analysis

**Open Stores:**
```
Predicted Sales (no COVID) = $1,000,000/month
Actual Sales (with COVID)  = $850,000/month
COVID Impact Factor        = -15%
```

**Closed Stores + Nearby Open Stores:**
```
Nearby Store Predicted (no COVID, no closure) = $1,000,000/month
Nearby Store Actual (with COVID, after closure) = $1,100,000/month
Combined Impact = +10% total
  = -15% (COVID) + 25% (halo effect)
```

#### Validation

Model quality assessed using:
- **MAPE (Mean Absolute Percentage Error)**: Average prediction error
- **Cross-validation**: Rolling time series validation
- **Visual inspection**: Comparing predicted vs. actual trends

---

### 3. Store Clustering & Mapping

For accurate halo effect prediction, stores are grouped by similarity:

**Clustering Dimensions:**
- Geographic region
- Store size (square footage)
- Sales volume tier
- Demographic profile of trade area
- Urban vs. suburban vs. rural

**Use Case:**
When Store A closes, find similar Store B to estimate expected halo effect based on historical Store B closure patterns.

---

### 4. Regression Modeling

The Jupyter notebook builds predictive models:

**Target Variable:** Closure Factor (sales lift percentage)

**Features:**
- Distance between closed and receiving store
- Trade area overlap percentage
- Pre-closure transaction volume
- Demographic similarity
- Store capacity (can it absorb additional customers?)
- Competitive landscape (other nearby options)

**Model Types Tested:**
- Linear regression (baseline)
- Random forest (captures non-linear relationships)
- Gradient boosting (best performance)

**Output:** Predicted closure factor for any store pair

---

## Data Sources & Requirements

### Input Data Files

Place these files in `data/raw/` directory:

#### BABY Store Analysis

| File | Description | Format | Size |
|------|-------------|--------|------|
| `BABY_Instore_Transactions_v2.pkl` | In-store purchase transactions by census block | Pandas DataFrame (pickle) | ~500MB |
| `BABY_Distance_data_full.pkl` | Haversine distances from census blocks to stores | Pandas DataFrame (pickle) | ~300MB |
| `BABY_Home_Store_full.pkl` | Customer home store preferences | Pandas DataFrame (pickle) | ~200MB |
| `BABY_BOPIS_v2.pkl` | Buy Online Pickup In Store orders | Pandas DataFrame (pickle) | ~150MB |

**Schema Example (Transactions):**
```python
{
    'store_nbr': int,               # Store identifier
    'FIPS_STATE_CD': str,           # State FIPS code (2 digits)
    'FIPS_COUNTY_CD': str,          # County FIPS code (3 digits)
    'CENSUS_BLOCK_CD': str,         # Census block code (6 digits)
    'instore_transaction_count': int # Number of transactions
}
```

#### BBBY Open Stores Analysis

| File | Description | Format | Size |
|------|-------------|--------|------|
| `BBBY_Open_Store_Transactions_v2.pkl` | In-store transactions (open stores) | Pandas DataFrame (pickle) | ~1.2GB |
| `BBBY_Open_Stores_Distance_data_full.pkl` | Distance data (open stores) | Pandas DataFrame (pickle) | ~800MB |
| `BBBY_open_store_BOPIS_v2.pkl` | BOPIS data (open stores) | Pandas DataFrame (pickle) | ~400MB |
| `BBBY_best_params_stores.csv` | Pre-computed DBSCAN parameters | CSV | ~50KB |

**Best Parameters Schema:**
```csv
Store_Number,Phase_1_eps_val,Phase_1_minsam,Phase_1_best_score
2001,0.5,5,0.67
2015,0.4,4,0.72
```

#### BBBY Closed Stores Analysis

| File | Description | Format | Size |
|------|-------------|--------|------|
| `BBBY_closed_Instore_Transactions_v2.pkl` | Transactions (closed stores, historical) | Pandas DataFrame (pickle) | ~600MB |
| `BBBY_closed_store_Distance_data_full.pkl` | Distance data (closed stores) | Pandas DataFrame (pickle) | ~400MB |
| `BBBY_closed_store_BOPIS_v2.pkl` | BOPIS data (closed stores) | Pandas DataFrame (pickle) | ~200MB |
| `closure_best_params_stores.csv` | Pre-computed parameters (closed stores) | CSV | ~30KB |

#### COVID-19 Forecasting

| File | Description | Format | Size |
|------|-------------|--------|------|
| `open_Stores_phase1_17_18_19_20_21.csv` | Monthly sales 2017-2021 (open stores) | CSV | ~5MB |
| `closed_Stores_phase1_17_18_19_20_21.csv` | Monthly sales 2017-2021 (closed stores) | CSV | ~2MB |
| `Store_Clusters.xlsx` | Store similarity clustering | Excel | ~500KB |

**Sales Data Schema:**
```csv
store_number,yearss,months,f0_
2001,2017,1,125000.50
2001,2017,2,118000.75
```

### External Data (Accessed via BigQuery)

**Census Bureau ACS (American Community Survey):**
- Dataset: `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`
- Fields used:
  - `geo_id`: Census block group identifier
  - `pop_25_years_over`: Population aged 25+ (retail target demographic)

**Access Requirements:**
- Google Cloud Platform account
- BigQuery API enabled
- Service account with BigQuery Data Viewer role
- `gcp_key.json` credentials file

---

## Installation & Setup

### Prerequisites

**System Requirements:**
- Python 3.7 or higher
- 16GB RAM (minimum) for large datasets
- 50GB free disk space for data files
- Internet connection for BigQuery access

**Required Accounts:**
- Google Cloud Platform account with BigQuery enabled
- GitHub account (for repository access)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/TanSin18/Halo-Effect.git
cd Halo-Effect
```

#### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
```
pandas>=1.3.0                    # Data manipulation
numpy>=1.21.0                    # Numerical computing
scikit-learn>=0.24.0             # Machine learning
prophet>=1.1.0                   # Time series forecasting
google-cloud-bigquery>=2.20.0    # GCP BigQuery access
haversine>=2.5.0                 # Distance calculations
matplotlib>=3.4.0                # Visualization
openpyxl>=3.0.7                  # Excel file support
python-dotenv>=0.19.0            # Environment variables
```

#### 4. Configure Environment

**Create configuration file:**
```bash
cp .env.example .env
nano .env  # or use your preferred text editor
```

**Edit `.env` with your settings:**
```bash
# Google Cloud Platform
GCP_KEY_PATH=/path/to/your/gcp_key.json
GCP_PROJECT_ID=your-project-id-123

# Data directories (optional, uses defaults if not set)
DATA_DIR=./data
OUTPUT_DIR=./output

# Processing configuration
NUM_WORKERS=50  # Adjust based on CPU cores
```

#### 5. Set Up Google Cloud Credentials

**Option A: Environment Variable (Recommended)**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp_key.json"
```

**Option B: In .env file**
```bash
GCP_KEY_PATH=/path/to/gcp_key.json
```

**Creating a GCP Service Account:**
1. Go to Google Cloud Console → IAM & Admin → Service Accounts
2. Click "Create Service Account"
3. Name: `halo-effect-bigquery`
4. Role: BigQuery Data Viewer
5. Create and download JSON key
6. Save as `config/gcp_key.json`

#### 6. Prepare Data Files

```bash
# Create data directory structure (already exists)
mkdir -p data/raw data/processed output/clusters output/params

# Copy your data files to data/raw/
cp /path/to/your/data/*.pkl data/raw/
cp /path/to/your/data/*.csv data/raw/
cp /path/to/your/data/*.xlsx data/raw/
```

#### 7. Verify Installation

```bash
# Test configuration
python -c "from config import config; config.print_config(); config.validate_config()"

# Should output:
# ============================================================
# Halo Effect Analysis - Configuration
# ============================================================
# Project Root: /path/to/Halo-Effect
# Data Directory: /path/to/Halo-Effect/data
# Output Directory: /path/to/Halo-Effect/output
# Number of Workers: 50
# GCP Project ID: your-project-id-123
# GCP Key Present: True
# ============================================================
# Configuration is valid!
```

#### 8. Optional: Install as Package

For system-wide access to command-line tools:

```bash
pip install -e .

# Now you can run from anywhere:
halo-baby      # BABY trade area analysis
halo-bbby      # BBBY trade area analysis
halo-closed    # Closed stores analysis
halo-forecast  # COVID forecasting
```

---

## Usage Guide

### Quick Start

**Run complete analysis pipeline:**

```bash
# 1. Identify BABY store trade areas (2-phase clustering)
python -m src.halo_effect.baby_trade_area

# 2. Identify BBBY open store trade areas (1-phase with refinement)
python -m src.halo_effect.bbby_trade_area

# 3. Analyze closed store trade areas
python -m src.halo_effect.closed_stores_trade_area

# 4. Forecast COVID-19 impact and halo effect
python -m src.halo_effect.forecasting_covid_halo
```

### Detailed Usage

#### 1. BABY Trade Area Analysis

**What it does:**
- Two-phase DBSCAN clustering to identify trade areas
- Hyperparameter optimization via grid search
- Outputs cluster assignments and best parameters

**Run:**
```bash
python -m src.halo_effect.baby_trade_area
```

**Expected Runtime:**
- ~100 stores: 15-30 minutes
- ~200 stores: 30-60 minutes
- Depends on CPU cores and data size

**Output:**
```
data/clusters/baby_{store_number}_clusters.csv
data/params/baby_{store_number}_params.csv
```

**Console Output:**
```
============================================================
BABY Trade Area Analysis - Starting
============================================================
Loading data files...
Data files loaded successfully
Loading census data from BigQuery...
Census data loaded: 220333 unique geo_ids
Fetching store list from BigQuery...
Processing 127 stores
Clustering started for: 2001
Initial transformation done for: 2001
Merging data done for: 2001
Scaled data for: 2001
First Iteration done for: 2001
Second Iteration done for: 2001
Clustering completed for: 2001
...
============================================================
Processing complete!
Time spent: 1245.67 seconds
Successful: 127/127
Failed: 0/127
Results saved to: /path/to/output/clusters
============================================================
```

#### 2. BBBY Open Store Trade Area Analysis

**What it does:**
- Uses pre-computed optimal parameters (faster)
- Single-phase DBSCAN with KMeans refinement
- Separates primary vs. secondary trade areas

**Run:**
```bash
python -m src.halo_effect.bbby_trade_area
```

**Expected Runtime:**
- ~500 stores: 30-45 minutes

**Output:**
```
output/clusters/bbby_{store_number}_clusters.csv
```

**Key Difference from BABY:**
- No parameter optimization (uses `BBBY_best_params_stores.csv`)
- Faster execution
- KMeans post-processing to identify core trade area

#### 3. Closed Store Trade Area Analysis

**What it does:**
- Analyzes historical trade areas for closed stores
- Helps understand customer redistribution patterns

**Run:**
```bash
python -m src.halo_effect.closed_stores_trade_area
```

**Expected Runtime:**
- ~50 closed stores: 10-15 minutes

**Output:**
```
output/clusters/closed_{store_number}_clusters.csv
```

**Use Case:**
Compare closed store trade areas with nearby open stores to identify overlap and predict halo effect magnitude.

#### 4. COVID-19 Impact Forecasting

**What it does:**
- Trains Prophet models on pre-pandemic data (2017-2019)
- Forecasts counterfactual 2020-2021 sales (without COVID)
- Compares predicted vs. actual to quantify COVID impact
- Processes both open and closed stores

**Run:**
```bash
python -m src.halo_effect.forecasting_covid_halo
```

**Expected Runtime:**
- ~500 stores: 60-90 minutes (Prophet is compute-intensive)

**Output:**
```
output/open_stores_phase1_predictions.csv
output/closed_stores_phase1_predictions.csv
output/covid_factor_calculation_phase1.csv
```

**Results Interpretation:**

```csv
Date,Predicted_gross_Sales,Actual_gross_Sales,Pct_Difference
2020-03-01,1000000,850000,-0.15  # COVID impact: -15%
2020-06-01,1000000,950000,-0.05  # Recovering: -5%
2020-12-01,1200000,1350000,0.125 # Holiday + halo: +12.5%
```

#### 5. Regression Modeling (Jupyter Notebook)

**Open the notebook:**
```bash
jupyter notebook notebooks/Closure_Factor_Regression_Model_BBBY.ipynb
```

**Workflow:**
1. Load trade area cluster results
2. Load transaction data and store attributes
3. Engineer features (distance, overlap, demographics)
4. Train regression models
5. Predict closure factors for store pairs
6. Validate against actual post-closure data

---

## Output & Results

### Cluster Assignment Files

**Location:** `output/clusters/`

**Format:** CSV files, one per store

**Filename Pattern:** `{brand}_{store_number}_clusters.csv`

**Example: `baby_2001_clusters.csv`**
```csv
FIPS_STATE_CD,FIPS_COUNTY_CD,CENSUS_BLOCK_CD,GEOID,Store Number
36,061,040200,36061040200,2001
36,061,040300,36061040300,2001
36,061,040400,36061040400,2001
```

**Fields:**
- `FIPS_STATE_CD`: State FIPS code (36 = New York)
- `FIPS_COUNTY_CD`: County FIPS code (061 = Manhattan)
- `CENSUS_BLOCK_CD`: Census block code (unique within county)
- `GEOID`: Combined identifier (concatenation of above)
- `Store Number`: Store identifier

**Usage:**
- Map trade areas geographically using GIS software (ArcGIS, QGIS)
- Calculate trade area overlap between stores
- Join with demographic data for customer profiling
- Feed into regression models as features

### Parameter Files

**Location:** `output/params/`

**Format:** CSV files, one per store

**Filename Pattern:** `{brand}_{store_number}_params.csv`

**Example: `baby_2001_params.csv`**
```csv
Phase_1_eps_val,Phase_1_minsam,Phase_1_best_score,Phase_2_eps_val,Phase_2_minsam,Phase_2_best_score,GEOID_1,instore_transaction_1,cust_store_dist_1,BOPIS_ORDER_counts_1,GEOID,instore_transaction,cust_store_dist,BOPIS_ORDER_counts
0.5,5,0.67,0.3,3,0.72,45,0.0023,8.5,0.0012,52,0.0025,7.8,0.0015
```

**Fields:**
- `Phase_1_eps_val`: Optimal DBSCAN epsilon for Phase 1
- `Phase_1_minsam`: Optimal minimum samples for Phase 1
- `Phase_1_best_score`: Silhouette score (quality metric)
- `Phase_2_*`: Same for Phase 2 refinement
- `GEOID_1`: Number of census blocks in Phase 1 cluster
- `GEOID`: Total census blocks in final trade area
- `instore_transaction`: Average transaction density
- `cust_store_dist`: Average customer distance (miles)
- `BOPIS_ORDER_counts`: Average BOPIS order density

**Usage:**
- Quality control (check silhouette scores)
- Compare store characteristics
- Reuse optimal parameters for similar stores
- Document methodology decisions

### Forecast Files

**Location:** `output/`

**Files:**
1. `open_stores_phase1_predictions.csv` - Open store forecasts
2. `closed_stores_phase1_predictions.csv` - Closed store forecasts
3. `covid_factor_calculation_phase1.csv` - Halo effect calculations

**Example: `open_stores_phase1_predictions.csv`**
```csv
Date,Predicted_gross_Sales,Predicted_Lower,Predicted_Upper,Actual Counts,store_number
2020-01-01,1000000,950000,1050000,980000,2001
2020-02-01,1050000,1000000,1100000,1030000,2001
2020-03-01,1000000,950000,1050000,750000,2001
```

**Fields:**
- `Date`: Month start date
- `Predicted_gross_Sales`: Point forecast (no COVID scenario)
- `Predicted_Lower`: 95% confidence interval lower bound
- `Predicted_Upper`: 95% confidence interval upper bound
- `Actual Counts`: Observed sales (with COVID)
- `store_number`: Store identifier

**Calculations:**
```python
# COVID impact
covid_impact = (Actual - Predicted) / Predicted

# Example: March 2020
covid_impact = (750000 - 1000000) / 1000000 = -25%
```

**Example: `covid_factor_calculation_phase1.csv`**
```csv
closed_store_number,DATE,open_store_number,Actual_gross_Sales,Predicted_gross_Sales,Pct_Difference
3001,2020-12-01,2001,1350000,1000000,0.35
3001,2020-12-01,2005,1150000,1050000,0.095
```

**Fields:**
- `closed_store_number`: Store that closed
- `DATE`: Month
- `open_store_number`: Nearby store that may benefit
- `Actual_gross_Sales`: Observed sales
- `Predicted_gross_Sales`: Expected without closure
- `Pct_Difference`: Halo effect estimate

**Interpretation:**
```
Store 2001: +35% lift
= -15% (COVID average) + 50% (halo from Store 3001 closure)

Store 2005: +9.5% lift
= -15% (COVID) + 24.5% (halo effect, smaller due to distance)
```

---

## Performance & Scalability

### Computational Requirements

**CPU:**
- Recommended: 8+ cores
- Uses multiprocessing: `NUM_WORKERS` parallel processes
- Default: 50 workers (reduce if limited cores)

**Memory:**
- Minimum: 16GB RAM
- Recommended: 32GB for large datasets
- Each worker process loads full dataset (memory intensive)

**Storage:**
- Data files: ~5GB (compressed pickles)
- Output files: ~500MB (cluster CSVs)
- Free space needed: 50GB (for temporary files)

### Execution Times

**Hardware: 16-core CPU, 32GB RAM, SSD**

| Task | Stores | Runtime | Bottleneck |
|------|--------|---------|------------|
| BABY Trade Area (2-phase) | 127 | 20 min | DBSCAN optimization |
| BBBY Open (1-phase) | 556 | 35 min | Data merging |
| Closed Stores | 48 | 12 min | DBSCAN |
| COVID Forecasting | 600 | 75 min | Prophet training |

### Optimization Tips

**1. Reduce Parallel Workers if Memory Limited:**
```python
# In config/config.py or .env
NUM_WORKERS = 20  # Instead of 50
```

**2. Process Stores in Batches:**
```python
# Modify main() in any module
store_list_batch = store_list[:100]  # First 100 stores
```

**3. Pre-compute Census Data:**
```python
# Save census data to avoid repeated BigQuery calls
census_data.to_pickle('data/processed/census_block_group_v2.pkl')
```

**4. Use SSD Storage:**
- Pickle file I/O is disk-intensive
- SSD provides 10x faster read/write than HDD

**5. Increase BigQuery Quota:**
- Default: 2000 queries per day
- Request increase if hitting limits

---

## Troubleshooting

### Common Errors

#### 1. ImportError: No module named 'fbprophet'

**Error:**
```
ImportError: No module named 'fbprophet'
```

**Solution:**
The `fbprophet` package was renamed to `prophet`. Update installation:
```bash
pip uninstall fbprophet
pip install prophet
```

---

#### 2. FileNotFoundError: GCP key file not found

**Error:**
```
FileNotFoundError: GCP key file not found at: /path/to/gcp_key.json
```

**Solution:**
Update `.env` file with correct path:
```bash
GCP_KEY_PATH=/absolute/path/to/your/gcp_key.json
```

Or set environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/gcp_key.json"
```

---

#### 3. Authentication Error with BigQuery

**Error:**
```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials.
```

**Solutions:**

**A. Check service account permissions:**
1. Go to GCP Console → IAM & Admin → IAM
2. Find your service account
3. Verify roles include: `BigQuery Data Viewer`

**B. Re-authenticate:**
```bash
gcloud auth application-default login
```

**C. Verify JSON key file is valid:**
```bash
cat config/gcp_key.json | jq .type
# Should output: "service_account"
```

---

#### 4. MemoryError during multiprocessing

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

**A. Reduce parallel workers:**
```python
# In .env file
NUM_WORKERS=10  # Reduce from 50
```

**B. Process in batches:**
```python
# Modify script to process stores in smaller groups
batch_size = 50
for i in range(0, len(store_list), batch_size):
    batch = store_list[i:i+batch_size]
    # Process batch
```

**C. Increase system swap:**
```bash
# Linux: Increase swap space
sudo fallocate -l 32G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

#### 5. Silhouette score warnings

**Warning:**
```
Warning: No valid clusters found for store 2001
```

**Cause:**
Store has insufficient data for clustering (too few transactions or geographic spread).

**Solutions:**

**A. Lower minimum sample threshold:**
```python
# In config.py
DBSCAN_MIN_SAMPLES_RANGE = (1, 5, 1)  # Instead of (1, 10, 1)
```

**B. Check data quality:**
```python
# Count transactions for problem store
df[df['store_nbr'] == 2001]['instore_transaction_count'].sum()
```

**C. Exclude stores below threshold:**
```python
# Filter stores with < 100 total transactions
min_transactions = 100
```

---

#### 6. Prophet installation fails

**Error on macOS:**
```
ERROR: Command errored out with exit status 1:
  command: python setup.py bdist_wheel
```

**Solution:**
Install dependencies first:
```bash
# macOS
brew install gcc

# Install cmdstan (Prophet dependency)
pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Then install prophet
pip install prophet
```

---

#### 7. Data file schema mismatch

**Error:**
```
KeyError: 'instore_transaction_count'
```

**Cause:**
Data file has different column names than expected.

**Solution:**
Check actual columns:
```python
import pandas as pd
df = pd.read_pickle('data/raw/BABY_Instore_Transactions_v2.pkl')
print(df.columns.tolist())
```

Update column names in script or rename columns in data:
```python
df.rename(columns={'trans_count': 'instore_transaction_count'}, inplace=True)
```

---

#### 8. BigQuery quota exceeded

**Error:**
```
google.api_core.exceptions.TooManyRequests: 429 Quota exceeded
```

**Solutions:**

**A. Cache census data:**
```python
# First run: save census data
census_data.to_pickle('data/processed/census_cache.pkl')

# Subsequent runs: load from cache
census_data = pd.read_pickle('data/processed/census_cache.pkl')
```

**B. Request quota increase:**
1. Go to GCP Console → IAM & Admin → Quotas
2. Find "BigQuery API - Queries per day"
3. Click "Edit Quotas" → Request increase

**C. Use batch processing:**
Spread analysis across multiple days to stay within daily limits.

---

## Contributing

We welcome contributions! Here's how to get involved:

### Ways to Contribute

1. **Report Bugs**: Open GitHub Issues with detailed reproduction steps
2. **Request Features**: Describe use cases and business value
3. **Improve Documentation**: Fix typos, add examples, clarify explanations
4. **Submit Code**: Add features, fix bugs, optimize performance

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Halo-Effect.git
cd Halo-Effect

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # Installs pytest, black, flake8

# Make changes and test
pytest tests/
black src/
flake8 src/

# Commit and push
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name

# Open Pull Request on GitHub
```

### Code Standards

**Python Style:**
- Follow PEP 8 guidelines
- Use `black` for automatic formatting
- Maximum line length: 100 characters
- Type hints for function signatures

**Documentation:**
- Docstrings for all public functions (Google style)
- Update README if adding features
- Include examples in docstrings

**Testing:**
- Write unit tests for new functions
- Maintain >80% code coverage
- Test edge cases (empty data, single store, etc.)

**Example:**
```python
def calculate_closure_factor(
    closed_store: int,
    open_store: int,
    trade_area_overlap: float
) -> float:
    """
    Calculate expected closure factor (halo effect magnitude).

    Args:
        closed_store: Store number of closed location
        open_store: Store number of receiving location
        trade_area_overlap: Percentage overlap of trade areas (0-1)

    Returns:
        Estimated closure factor (e.g., 0.15 = 15% sales increase)

    Example:
        >>> calculate_closure_factor(3001, 2001, 0.45)
        0.127  # 12.7% expected lift
    """
    # Implementation
    pass
```

---

## License

MIT License

Copyright (c) 2021 Tanmay Sinnarkar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Authors

**Tanmay Sinnarkar** - Original Analysis & Development
- GitHub: [@TanSin18](https://github.com/TanSin18)
- Project: Halo Effect Analysis for Retail Store Closures
- Year: 2021

### Acknowledgments

- **Bed Bath & Beyond / Buy Buy Baby**: Data source for retail analysis
- **Google BigQuery**: Census data access
- **Facebook Prophet Team**: Time series forecasting framework
- **scikit-learn Community**: Machine learning tools

---

## References

### Academic Papers

1. **DBSCAN Algorithm:**
   - Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD-96 Proceedings*.

2. **Retail Trade Areas:**
   - Huff, D.L. (1963). "A Probabilistic Analysis of Shopping Center Trade Areas." *Land Economics*, 39(1), 81-90.

3. **Time Series Forecasting:**
   - Taylor, S.J., & Letham, B. (2018). "Forecasting at scale." *The American Statistician*, 72(1), 37-45.

### Documentation & Tools

- [DBSCAN Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) - scikit-learn documentation
- [Prophet Forecasting](https://facebook.github.io/prophet/) - Official Prophet documentation
- [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data) - Google Cloud documentation
- [Census Bureau ACS](https://www.census.gov/programs-surveys/acs) - American Community Survey data

### Related Projects

- [Retail Analytics Toolkit](https://github.com/retail-analytics) - Broader retail analysis tools
- [Store Locator Optimization](https://github.com/store-optimization) - Store location planning
- [Customer Segmentation](https://github.com/customer-segments) - Demographic clustering

---

## Citation

If you use this work in your research or business applications, please cite:

**BibTeX:**
```bibtex
@software{halo_effect_2021,
  author = {Sinnarkar, Tanmay},
  title = {Halo Effect Analysis: Retail Store Closure Impact Quantification Framework},
  year = {2021},
  publisher = {GitHub},
  url = {https://github.com/TanSin18/Halo-Effect},
  version = {1.0.0}
}
```

**APA Style:**
```
Sinnarkar, T. (2021). Halo Effect Analysis: Retail Store Closure Impact Quantification Framework
(Version 1.0.0) [Computer software]. GitHub. https://github.com/TanSin18/Halo-Effect
```

**Chicago Style:**
```
Sinnarkar, Tanmay. 2021. "Halo Effect Analysis: Retail Store Closure Impact Quantification
Framework." Version 1.0.0. GitHub. https://github.com/TanSin18/Halo-Effect.
```

---

## Contact & Support

**Questions?** Open a GitHub Issue or Discussion

**Bug Reports:** Use GitHub Issues with the `bug` label

**Feature Requests:** Use GitHub Issues with the `enhancement` label

**General Inquiries:** Contact via GitHub profile

---

## Version History

### v1.0.0 (2021)
- Initial release
- DBSCAN trade area clustering for BABY and BBBY stores
- Two-phase optimization with automatic parameter tuning
- Prophet-based COVID-19 impact forecasting
- Closed store trade area analysis
- BigQuery census data integration
- Multiprocessing support for scalability
- Comprehensive documentation

---

**Last Updated:** 2021
**Maintained By:** Tanmay Sinnarkar
**Status:** Active Development
