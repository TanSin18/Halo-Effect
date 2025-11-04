# Halo Effect Analysis

A comprehensive analytical framework for measuring the "halo effect" of retail store closures on surrounding stores using machine learning clustering techniques and time series forecasting.

## Overview

This project analyzes the impact of store closures on nearby retail locations by:
- Identifying trade areas using DBSCAN clustering
- Forecasting COVID-19 impact using Prophet time series models
- Building regression models to predict closure factors
- Analyzing both Bed Bath & Beyond (BBBY) and Buy Buy Baby (BABY) stores

## Project Structure

```
Halo-Effect/
├── src/
│   └── halo_effect/
│       ├── __init__.py
│       ├── baby_trade_area.py      # Trade area analysis for BABY stores
│       ├── bbby_trade_area.py      # Trade area analysis for BBBY stores
│       ├── closed_stores_trade_area.py  # Analysis for closed stores
│       └── forecasting_covid_halo.py    # COVID-19 impact forecasting
├── data/
│   ├── raw/                        # Raw data files (pickles, CSVs)
│   └── processed/                  # Processed data outputs
├── output/
│   ├── clusters/                   # Cluster analysis results
│   └── params/                     # Model parameters
├── notebooks/
│   └── Closure_Factor_Regression_Model_BBBY.ipynb
├── config/
│   ├── config.py                   # Configuration management
│   └── .env.example                # Environment variables template
├── requirements.txt
├── setup.py
└── README.md
```

## Features

### Trade Area Identification
- **DBSCAN Clustering**: Identifies store trade areas using multi-dimensional clustering
- **Two-Phase Clustering**: Initial broad clustering followed by refined segmentation
- **Hyperparameter Optimization**: Automatic tuning using silhouette scores
- **Census Integration**: Normalizes transaction data by population density

### COVID-19 Impact Forecasting
- **Prophet Time Series**: Forecasts store performance with multiplicative seasonality
- **Comparative Analysis**: Compares open vs closed stores to isolate COVID impact
- **Store Clustering**: Groups similar stores for more accurate predictions

### Data Sources
- In-store transaction data
- BOPIS (Buy Online Pickup In Store) data
- Distance/geolocation data (census block level)
- Google BigQuery census data
- Historical sales data

## Installation

### Prerequisites
- Python 3.7+
- Google Cloud Platform account with BigQuery access
- GCP service account key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TanSin18/Halo-Effect.git
cd Halo-Effect
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp config/.env.example .env
# Edit .env with your settings
```

5. Set up Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/gcp_key.json"
```

## Configuration

Edit `config/config.py` or set environment variables:

```python
# Required
GCP_KEY_PATH = "/path/to/gcp_key.json"
PROJECT_ID = "your-gcp-project-id"

# Optional
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
NUM_WORKERS = 50  # Multiprocessing pool size
```

## Usage

### Trade Area Analysis (BABY Stores)

```bash
python -m src.halo_effect.baby_trade_area
```

Features:
- Loads transaction, distance, and BOPIS data
- Performs two-phase DBSCAN clustering
- Outputs cluster assignments and best parameters
- Uses multiprocessing for parallel execution

### Trade Area Analysis (BBBY Open Stores)

```bash
python -m src.halo_effect.bbby_trade_area
```

Features:
- Uses pre-computed best parameters
- Single-phase clustering with KMeans refinement
- Identifies primary trade areas

### Closed Stores Analysis

```bash
python -m src.halo_effect.closed_stores_trade_area
```

Analyzes trade areas for closed stores to understand customer redistribution.

### COVID-19 Halo Effect Forecasting

```bash
python -m src.halo_effect.forecasting_covid_halo
```

Features:
- Prophet forecasting models for open and closed stores
- Calculates COVID impact factor
- Compares predicted vs actual performance
- Stores cluster-based mapping

## Data Requirements

Place the following files in `data/raw/`:

### BABY Analysis
- `BABY_Instore_Transactions_v2.pkl`
- `BABY_Distance_data_full.pkl`
- `BABY_Home_Store_full.pkl`
- `BABY_BOPIS_v2.pkl`

### BBBY Analysis
- `BBBY_Open_Store_Transactions_v2.pkl`
- `BBBY_Open_Stores_Distance_data_full.pkl`
- `BBBY_open_store_BOPIS_v2.pkl`
- `BBBY_best_params_stores.csv`

### Closed Stores Analysis
- `BBBY_closed_Instore_Transactions_v2.pkl`
- `BBBY_closed_store_Distance_data_full.pkl`
- `BBBY_closed_store_BOPIS_v2.pkl`
- `closure_best_params_stores.csv`

### Forecasting
- `open_Stores_phase1_17_18_19_20_21.csv`
- `closed_Stores_phase1_17_18_19_20_21.csv`
- `Store_Clusters.xlsx`

## Methodology

### DBSCAN Clustering Algorithm

1. **Data Normalization**: Transaction counts normalized by census block population
2. **Feature Scaling**: StandardScaler applied to all features
3. **Parameter Grid Search**: Tests multiple eps and min_samples combinations
4. **Silhouette Optimization**: Selects parameters maximizing silhouette score
5. **Two-Phase Approach**:
   - Phase 1: Broad clustering (all stores)
   - Phase 2: Refine cluster 0 (outliers/mixed areas)

### Features Used
- In-store transaction density (normalized by population)
- Customer-to-store distance
- BOPIS order density
- Distance transformation: `50 - distance` (inverted for clustering)

### Prophet Forecasting

- **Seasonality Mode**: Multiplicative (better for retail with seasonal variations)
- **Changepoint Prior**: 0.001 (conservative to avoid overfitting)
- **Training Data**: 2017-2019 historical data
- **Test Period**: 2020-2021 (includes COVID period)
- **Validation**: Mean Absolute Percentage Error (MAPE)

## Output Files

### Cluster Results
- `output/clusters/{store_number}_clusters.csv`: Census block to store mapping

### Parameters
- `output/params/{store_number}_params.csv`: Best DBSCAN parameters and metrics

### Forecasts
- `output/covid_factor_calculation_phase1.csv`: COVID impact factors
- `output/closed_stores_phase1_predictions.csv`: Closed store forecasts

## Performance

- **Multiprocessing**: Utilizes 50-60 parallel workers
- **Processing Time**: Varies by store count (typically 100-500 stores)
- **Memory**: Requires significant RAM for large datasets (recommend 16GB+)

## Database Queries

The `Regression_Model_Data.sql` file contains queries for:
- Creating store-customer mappings from trade area clusters
- Extracting transaction history
- Joining census and demographic data
- Building regression model datasets

## Dependencies

Key libraries:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: DBSCAN, StandardScaler, KMeans
- `prophet` (formerly fbprophet): Time series forecasting
- `google-cloud-bigquery`: GCP data access
- `haversine`: Distance calculations
- `matplotlib`: Visualization

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

**ImportError: No module named 'fbprophet'**
- Solution: Install `prophet` (renamed package): `pip install prophet`

**Authentication Error with BigQuery**
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
- Verify service account has BigQuery Data Viewer permissions

**MemoryError during multiprocessing**
- Reduce `NUM_WORKERS` in config
- Process stores in batches

**Silhouette score warnings**
- Normal for stores with insufficient data
- Check if store has minimum transaction threshold

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Authors

- Original Analysis: SK0759
- Repository: TanSin18

## References

- [DBSCAN Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Prophet Forecasting](https://facebook.github.io/prophet/)
- [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{halo_effect_2021,
  author = {SK0759},
  title = {Halo Effect Analysis: Retail Store Closure Impact},
  year = {2021},
  url = {https://github.com/TanSin18/Halo-Effect}
}
```
