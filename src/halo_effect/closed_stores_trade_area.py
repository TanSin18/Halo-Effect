"""
Closed Stores Trade Area Analysis

This module performs trade area identification for closed Bed Bath & Beyond stores
using pre-computed DBSCAN parameters and KMeans refinement.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud import bigquery_storage

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


def init_bigquery_client():
    """Initialize BigQuery client with credentials"""
    if not Path(config.GCP_KEY_PATH).exists():
        raise FileNotFoundError(
            f"GCP key file not found at: {config.GCP_KEY_PATH}\n"
            "Please update .env file or set GCP_KEY_PATH environment variable"
        )

    credentials = service_account.Credentials.from_service_account_file(
        config.GCP_KEY_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    bq_client = bigquery.Client(
        credentials=credentials,
        project=credentials.project_id
    )
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)
    job_config = bigquery.QueryJobConfig(allow_large_results=config.BIGQUERY_ALLOW_LARGE_RESULTS)

    return bq_client, bqstorageclient, job_config


def load_data():
    """Load all required data files"""
    print("Loading data files...")

    data_files = {
        'transactions': config.BBBY_CLOSED_TRANSACTION_DATA,
        'distance': config.BBBY_CLOSED_DISTANCE_DATA,
        'bopis': config.BBBY_CLOSED_BOPIS_DATA,
        'best_params': config.CLOSURE_BEST_PARAMS,
    }

    # Check if files exist
    for name, filepath in data_files.items():
        if not filepath.exists():
            raise FileNotFoundError(
                f"{name.capitalize()} data file not found at: {filepath}\n"
                f"Please place the required data files in {config.RAW_DATA_DIR}"
            )

    # Load data
    data = {
        'transactions': pd.read_pickle(str(config.BBBY_CLOSED_TRANSACTION_DATA)),
        'distance': pd.read_pickle(str(config.BBBY_CLOSED_DISTANCE_DATA)),
        'bopis': pd.read_pickle(str(config.BBBY_CLOSED_BOPIS_DATA)),
        'best_params': pd.read_csv(str(config.CLOSURE_BEST_PARAMS)),
    }

    print("Data files loaded successfully")
    return data


def load_census_data(bq_client, job_config):
    """Load and process census data from BigQuery"""
    print("Loading census data from BigQuery...")

    census_block_group = bq_client.query(
        config.CENSUS_QUERY,
        job_config=job_config
    ).result().to_dataframe()

    census_block_group['geo_id'] = census_block_group['geo_id'].apply(lambda x: str(x)[:-1])
    census_block_group_v2 = census_block_group.groupby(['geo_id'])['pop_25_years_over'].sum().reset_index()

    print(f"Census data loaded: {census_block_group_v2['geo_id'].nunique()} unique geo_ids")
    return census_block_group_v2


def clustering_func(store_number, data, census_block_group_v2):
    """
    Perform single-phase DBSCAN clustering with KMeans refinement for a closed store

    Args:
        store_number: Store number to process
        data: Dictionary containing transactions, distance, bopis, and best_params DataFrames
        census_block_group_v2: Census data DataFrame

    Returns:
        0 on success
    """
    try:
        print(f"Clustering started for: {store_number}")

        # Get pre-computed parameters
        store_params_v2 = data['best_params'][
            data['best_params']['Store_Number'] == int(store_number)
        ]

        if len(store_params_v2) == 0:
            print(f"Warning: No parameters found for store {store_number}, skipping")
            return 1

        # Filter data for this store
        Instore_Transaction_data = data['transactions'][
            data['transactions']['store_nbr'] == int(store_number)
        ]
        Distance_data = data['distance'][
            data['distance']['store_nbr'] == int(store_number)
        ]
        BOPIS = data['bopis'][
            data['bopis']['PICKUP_STORE'] == int(store_number)
        ]

        # Standardize FIPS codes
        for df in [Instore_Transaction_data, Distance_data, BOPIS]:
            df['FIPS_STATE_CD'] = df['FIPS_STATE_CD'].apply(
                lambda x: str(x).zfill(2) if len(str(x))!=2 else str(x)
            )
            df['FIPS_COUNTY_CD'] = df['FIPS_COUNTY_CD'].apply(
                lambda x: str(x).zfill(3) if len(str(x))!=3 else str(x)
            )
            df['CENSUS_BLOCK_CD'] = df['CENSUS_BLOCK_CD'].apply(
                lambda x: str(x).zfill(6) if len(str(x))!=6 else str(x)
            )
            df['GEOID'] = df[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)

        # Merge with census data and normalize
        Instore_Transaction_data_v2 = Instore_Transaction_data.merge(
            census_block_group_v2, how='left', left_on='GEOID', right_on='geo_id'
        )
        Instore_Transaction_data_v2['instore_transaction_count_v2'] = (
            Instore_Transaction_data_v2['instore_transaction_count'] /
            Instore_Transaction_data_v2['pop_25_years_over']
        )

        BOPIS_v2 = BOPIS.merge(
            census_block_group_v2, how='left', left_on='GEOID', right_on='geo_id'
        )
        BOPIS_v2['BOPIS_Counts'] = (
            BOPIS_v2['CUSTOMER_ID'] / BOPIS_v2['pop_25_years_over']
        )

        print(f"Initial transformation done for: {store_number}")

        # Merge all data
        ITD_DD_HS = Instore_Transaction_data_v2.merge(
            Distance_data, how='outer',
            left_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD'],
            right_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']
        )
        ITD_DD_HS_BP = ITD_DD_HS.merge(
            BOPIS_v2, how='outer',
            left_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD'],
            right_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']
        )

        ITD_DD_HS_BP.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"Merging data done for: {store_number}")

        # Handle missing values
        ITD_DD_HS_BP['instore_transaction_count_v2'] = (
            ITD_DD_HS_BP['instore_transaction_count_v2'].fillna(0)
        )
        ITD_DD_HS_BP['distance'] = ITD_DD_HS_BP['distance'].fillna(config.MAX_DISTANCE)
        ITD_DD_HS_BP['distance_v2'] = config.MAX_DISTANCE - ITD_DD_HS_BP['distance']
        ITD_DD_HS_BP['BOPIS_Counts'] = ITD_DD_HS_BP['BOPIS_Counts'].fillna(0)

        print(f"Unknown data treated for: {store_number}")

        # Scale features
        BOPIS_Scaler = StandardScaler()
        ITD_DD_HS_BP[['BOPIS_Counts_scaled']] = BOPIS_Scaler.fit_transform(
            ITD_DD_HS_BP[['BOPIS_Counts']]
        )

        Instore_Transactions_Scaler = StandardScaler()
        ITD_DD_HS_BP[['instore_transaction_count_v2_scaled']] = (
            Instore_Transactions_Scaler.fit_transform(
                ITD_DD_HS_BP[['instore_transaction_count_v2']]
            )
        )

        Distance_Scaler = StandardScaler()
        ITD_DD_HS_BP[['cust_store_dist_scaled']] = Distance_Scaler.fit_transform(
            ITD_DD_HS_BP[['distance_v2']]
        )

        print(f"Scaled data for: {store_number}")

        # Prepare features for clustering
        X = ITD_DD_HS_BP[[
            'instore_transaction_count_v2_scaled',
            'cust_store_dist_scaled',
            'BOPIS_Counts_scaled'
        ]]

        # Apply DBSCAN with pre-computed parameters
        clustering = DBSCAN(
            eps=float(store_params_v2['Phase_1_eps_val'].iloc[0]),
            min_samples=int(store_params_v2['Phase_1_minsam'].iloc[0])
        ).fit(X)

        clusters = ITD_DD_HS_BP.copy()
        clusters['DBScan_Clusters'] = clustering.labels_

        print(f"First Iteration done for: {store_number}")

        # Inverse transform scaled features
        clusters[['instore_transaction']] = Instore_Transactions_Scaler.inverse_transform(
            clusters[['instore_transaction_count_v2_scaled']]
        )
        clusters[['cust_store_dist']] = Distance_Scaler.inverse_transform(
            clusters[['cust_store_dist_scaled']]
        )
        clusters[['BOPIS_ORDER_counts']] = BOPIS_Scaler.inverse_transform(
            clusters[['BOPIS_Counts_scaled']]
        )

        # Exclude noise cluster
        clusters = clusters[clusters['DBScan_Clusters'] != 0]

        clusters['GEOID'] = clusters[[
            'FIPS_STATE_CD', 'FIPS_COUNTY_CD', 'CENSUS_BLOCK_CD'
        ]].agg(''.join, axis=1)

        clusters_final = clusters.copy()
        clusters_final['Store Number'] = [store_number] * len(clusters_final)
        clusters_final = clusters_final[clusters_final['distance'] < config.MAX_DISTANCE]

        # KMeans refinement to identify primary trade area
        if len(clusters_final) > 2:
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans_x = clusters_final[['distance']]
            kmeans.fit(kmeans_x)
            clusters_final['kmeans_cluster'] = kmeans.labels_

            # Select cluster with smallest average distance
            clus_num = clusters_final.groupby(['kmeans_cluster'])['distance'].mean()\
                .reset_index()\
                .sort_values(['distance'], ascending=True)\
                .head(1)['kmeans_cluster']

            clusters_final_v2 = clusters_final[
                clusters_final['kmeans_cluster'].isin(list(clus_num))
            ]
        else:
            clusters_final_v2 = clusters_final

        # Save cluster results
        output_file = config.CLUSTERS_OUTPUT_DIR / f"closed_{store_number}_clusters.csv"
        clusters_final_v2[[
            'FIPS_STATE_CD', 'FIPS_COUNTY_CD', 'CENSUS_BLOCK_CD',
            'GEOID', 'Store Number'
        ]].drop_duplicates().to_csv(output_file, index=False)

        print(f"Cluster data output for: {store_number}")
        print(f"Clustering completed for: {store_number}")

        return 0

    except Exception as e:
        print(f"Error processing store {store_number}: {str(e)}")
        return 1


def main():
    """Main execution function"""
    print("=" * 60)
    print("BBBY Closed Stores Trade Area Analysis - Starting")
    print("=" * 60)

    # Validate configuration
    if not config.validate_config():
        print("\nPlease fix configuration issues before running.")
        return 1

    try:
        # Initialize BigQuery
        bq_client, bqstorageclient, job_config = init_bigquery_client()

        # Load data
        data = load_data()
        census_data = load_census_data(bq_client, job_config)

        # Get store list
        print("Fetching store list from BigQuery...")
        closed_stores = bq_client.query(
            config.BBBY_CLOSED_STORES_QUERY,
            job_config=job_config
        ).result().to_dataframe()

        store_list = list(closed_stores['Store__'].unique())
        print(f"Processing {len(store_list)} stores")

        # Process stores in parallel
        start = time.time()
        with Pool(config.NUM_WORKERS) as pool:
            results = pool.starmap(
                clustering_func,
                [(store, data, census_data) for store in store_list]
            )
        end = time.time()

        print("=" * 60)
        print(f"Processing complete!")
        print(f"Time spent: {end-start:.2f} seconds")
        print(f"Successful: {results.count(0)}/{len(results)}")
        print(f"Failed: {results.count(1)}/{len(results)}")
        print(f"Results saved to: {config.CLUSTERS_OUTPUT_DIR}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
