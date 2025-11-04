"""
COVID-19 Halo Effect Forecasting

This module uses Prophet time series forecasting to predict the impact of COVID-19
on store sales and identify the halo effect of closed stores on nearby open stores.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from prophet import Prophet  # Updated from fbprophet to prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn import metrics

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


def forecast_model(df, periods=18):
    """
    Forecast future sales using Prophet

    Args:
        df: DataFrame with 'ds' (date) and 'y' (value) columns
        periods: Number of months to forecast

    Returns:
        DataFrame with forecasted values
    """
    print("Training Prophet model...")

    # Fit Prophet model
    m = Prophet(
        seasonality_mode=config.PROPHET_SEASONALITY_MODE,
        changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR_SCALE
    )
    m.fit(df)

    # Forecast
    future = m.make_future_dataframe(periods=periods, freq=config.FORECAST_FREQ)
    forecast = m.predict(future)

    # Extract results
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    results['actual'] = df[-periods:]['y'].values if len(df) >= periods else np.nan
    results.columns = [
        'Date', 'Predicted_gross_Sales', 'Predicted_Lower',
        'Predicted_Upper', 'Actual Counts'
    ]

    return results


def process_open_stores():
    """Process and forecast open stores data"""
    print("=" * 60)
    print("Processing Open Stores")
    print("=" * 60)

    if not config.OPEN_STORES_DATA.exists():
        raise FileNotFoundError(
            f"Open stores data not found at: {config.OPEN_STORES_DATA}\n"
            f"Please place the file in {config.RAW_DATA_DIR}"
        )

    open_df = pd.read_csv(str(config.OPEN_STORES_DATA))
    open_df['months'] = open_df['months'].apply(lambda x: str(x).zfill(2))

    store_number = []
    MAPE_score = []
    predictions = []

    for store in list(open_df['store_number'].unique()):
        print(f"Processing open store: {store}")

        open_df_1 = open_df[open_df['store_number'] == store]
        open_df_1['DATE'] = open_df_1.apply(
            lambda x: f"{x['yearss']}-{x['months']}-01", axis=1
        )

        # Split train/test
        df_train = open_df_1[
            (open_df_1['yearss'] != 2021) & (open_df_1['yearss'] != 2020)
        ]
        df_test = open_df_1[
            (open_df_1['yearss'] == 2021) | (open_df_1['yearss'] == 2020)
        ]

        # Prepare training data
        df_train = df_train[['DATE', 'f0_']]
        df_train = df_train.rename(columns={'DATE': 'ds', 'f0_': 'y'})

        # Forecast
        df_train_results = forecast_model(df_train, periods=config.FORECAST_PERIODS)

        # Merge with test data
        df_test['DATE'] = df_test['DATE'].astype('datetime64[ns]')
        data_merged = df_train_results.merge(
            df_test, how='inner', left_on='Date', right_on='DATE'
        )

        predictions.append(data_merged)
        store_number.append(store)

        # Calculate MAPE
        mape = (
            metrics.mean_absolute_error(
                data_merged['f0_'],
                data_merged['Predicted_gross_Sales']
            ) / np.mean(data_merged['f0_']) * 100
        )
        MAPE_score.append(mape)

    # Combine all predictions
    open_result_data = pd.concat(predictions, ignore_index=True)
    open_result_data.rename(columns={'f0_': 'Actual_gross_Sales'}, inplace=True)
    open_result_data['Pct_Difference'] = (
        (open_result_data['Actual_gross_Sales'] /
         open_result_data['Predicted_gross_Sales']) - 1
    )

    print(f"Processed {len(store_number)} open stores")
    print(f"Average MAPE: {np.mean(MAPE_score):.2f}%")

    return open_result_data


def process_closed_stores():
    """Process and forecast closed stores data"""
    print("=" * 60)
    print("Processing Closed Stores")
    print("=" * 60)

    if not config.CLOSED_STORES_DATA.exists():
        raise FileNotFoundError(
            f"Closed stores data not found at: {config.CLOSED_STORES_DATA}\n"
            f"Please place the file in {config.RAW_DATA_DIR}"
        )

    df = pd.read_csv(str(config.CLOSED_STORES_DATA))
    df['months'] = df['months'].apply(lambda x: str(x).zfill(2))

    closed_store_number = []
    closed_MAPE_score = []
    closed_predictions = []

    for store in list(df['store_number'].unique()):
        print(f"Processing closed store: {store}")

        df_1 = df[df['store_number'] == store]
        df_1['DATE'] = df_1.apply(
            lambda x: f"{x['yearss']}-{x['months']}-01", axis=1
        )

        # Split train/test
        df_train = df_1[
            (df_1['yearss'] != 2021) & (df_1['yearss'] != 2020)
        ]
        df_test = df_1[
            (df_1['yearss'] == 2021) | (df_1['yearss'] == 2020)
        ]

        # Prepare training data
        df_train = df_train[['DATE', 'f0_']]
        df_train = df_train.rename(columns={'DATE': 'ds', 'f0_': 'y'})

        # Forecast
        df_train_results = forecast_model(df_train, periods=config.FORECAST_PERIODS)

        # Merge with test data
        df_test['DATE'] = df_test['DATE'].astype('datetime64[ns]')
        data_merged = df_train_results.merge(
            df_test, how='inner', left_on='Date', right_on='DATE'
        )

        closed_predictions.append(data_merged)
        closed_store_number.append(store)

        # Calculate MAPE
        mape = (
            metrics.mean_absolute_error(
                data_merged['f0_'],
                data_merged['Predicted_gross_Sales']
            ) / np.mean(data_merged['f0_']) * 100
        )
        closed_MAPE_score.append(mape)

    # Combine all predictions
    closed_result_data = pd.concat(closed_predictions, ignore_index=True)
    closed_result_data.rename(columns={'f0_': 'Actual_gross_Sales'}, inplace=True)

    print(f"Processed {len(closed_store_number)} closed stores")
    print(f"Average MAPE: {np.mean(closed_MAPE_score):.2f}%")

    # Save closed store predictions
    output_file = config.OUTPUT_DIR / 'closed_stores_phase1_predictions.csv'
    closed_result_data.to_csv(output_file, index=False)
    print(f"Saved closed store predictions to: {output_file}")

    return closed_result_data


def calculate_halo_effect(open_result_data):
    """
    Calculate halo effect by mapping closed to open stores

    Args:
        open_result_data: DataFrame with open store forecasts

    Returns:
        DataFrame with halo effect calculations
    """
    print("=" * 60)
    print("Calculating Halo Effect")
    print("=" * 60)

    if not config.STORE_CLUSTERS_DATA.exists():
        print(f"Warning: Store clusters data not found at: {config.STORE_CLUSTERS_DATA}")
        print("Skipping halo effect calculation")
        return None

    # Load store clusters
    store_clusters = pd.read_excel(
        str(config.STORE_CLUSTERS_DATA),
        sheet_name='Store_clusters2019Fiscal'
    )
    store_clusters.fillna('UNKNOWN', inplace=True)

    # Note: This section requires additional data (closed_Store_factor, open_Store_factor)
    # that is not in the original data files. Including the mapping logic structure.

    print("Store clusters loaded")
    print("Note: Complete halo effect calculation requires additional data files")
    print("      (closed_Store_factor, open_Store_factor)")

    # Placeholder for complete implementation
    # Would include:
    # 1. Map closed stores to similar open stores by cluster
    # 2. Aggregate open store performance by closed store
    # 3. Calculate COVID impact factor
    # 4. Save results

    return None


def main():
    """Main execution function"""
    print("=" * 60)
    print("COVID-19 Halo Effect Forecasting - Starting")
    print("=" * 60)

    try:
        # Process open stores
        open_result_data = process_open_stores()

        # Process closed stores
        closed_result_data = process_closed_stores()

        # Calculate halo effect (if data available)
        halo_effect_data = calculate_halo_effect(open_result_data)

        # Save open store results
        output_file = config.OUTPUT_DIR / 'open_stores_phase1_predictions.csv'
        open_result_data.to_csv(output_file, index=False)
        print(f"Saved open store predictions to: {output_file}")

        print("=" * 60)
        print("Forecasting complete!")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
