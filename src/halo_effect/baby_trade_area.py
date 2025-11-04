"""
BABY Trade Area Analysis

This module performs trade area identification for Buy Buy Baby stores using
DBSCAN clustering with two-phase optimization.
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
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
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
        'transactions': config.BABY_TRANSACTION_DATA,
        'distance': config.BABY_DISTANCE_DATA,
        'home_store': config.BABY_HOME_STORE_DATA,
        'bopis': config.BABY_BOPIS_DATA,
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
        'transactions': pd.read_pickle(str(config.BABY_TRANSACTION_DATA)),
        'distance': pd.read_pickle(str(config.BABY_DISTANCE_DATA)),
        'home_store': pd.read_pickle(str(config.BABY_HOME_STORE_DATA)),
        'bopis': pd.read_pickle(str(config.BABY_BOPIS_DATA)),
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
    Perform two-phase DBSCAN clustering for a single store

    Args:
        store_number: Store number to process
        data: Dictionary containing transactions, distance, home_store, and bopis DataFrames
        census_block_group_v2: Census data DataFrame

    Returns:
        0 on success
    """
    try:
        print(f"Clustering started for: {store_number}")

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
            'BOPIS_Counts_scaled',
            'instore_transaction_count_v2_scaled',
            'cust_store_dist_scaled'
        ]]

        # Phase 1: Find best parameters
        eps_list = np.arange(*config.DBSCAN_EPS_RANGE)
        min_sample_list = np.arange(*config.DBSCAN_MIN_SAMPLES_RANGE)

        silhouette_scores_data = pd.DataFrame()
        for eps_trial in eps_list:
            for min_sample_trial in min_sample_list:
                db = DBSCAN(eps=eps_trial, min_samples=int(min_sample_trial))

                if len(np.unique(db.fit_predict(X))) > 1:
                    sil_score = silhouette_score(X, db.fit_predict(X))
                else:
                    continue

                eps_parameter = str(eps_trial.round(1))
                min_sample_parameter = str(int(min_sample_trial))

                silhouette_scores_data = pd.concat([
                    silhouette_scores_data,
                    pd.DataFrame(
                        data=[[sil_score, eps_parameter, min_sample_parameter]],
                        columns=["score", "eps", "min_sample"]
                    )
                ], ignore_index=True)

        best_values = silhouette_scores_data.sort_values(
            by='score', ascending=False
        ).head(1)

        eps_val = list(best_values['eps'])[0]
        minsam = list(best_values['min_sample'])[0]
        best_score = list(best_values['score'])[0]

        clustering = DBSCAN(eps=float(eps_val), min_samples=int(minsam)).fit(X)
        clusters = ITD_DD_HS_BP.copy()
        clusters['DBScan_Clusters'] = clustering.labels_

        print(f"First Iteration done for: {store_number}")

        # Phase 2: Refine cluster 0
        data_phase_2 = clusters[clusters['DBScan_Clusters'] == 0]
        X_2 = data_phase_2[[
            'BOPIS_Counts_scaled',
            'instore_transaction_count_v2_scaled',
            'cust_store_dist_scaled'
        ]]

        silhouette_scores_data_2 = pd.DataFrame()
        for eps_trial in eps_list:
            for min_sample_trial in min_sample_list:
                db = DBSCAN(eps=eps_trial, min_samples=int(min_sample_trial))

                if len(np.unique(db.fit_predict(X_2))) > 1:
                    sil_score = silhouette_score(X_2, db.fit_predict(X_2))
                else:
                    continue

                eps_parameter = str(eps_trial.round(1))
                min_sample_parameter = str(int(min_sample_trial))

                silhouette_scores_data_2 = pd.concat([
                    silhouette_scores_data_2,
                    pd.DataFrame(
                        data=[[sil_score, eps_parameter, min_sample_parameter]],
                        columns=["score", "eps", "min_sample"]
                    )
                ], ignore_index=True)

        best_values_2 = silhouette_scores_data_2.sort_values(
            by='score', ascending=False
        ).head(1)

        eps_val_2 = list(best_values_2['eps'])[0]
        minsam_2 = list(best_values_2['min_sample'])[0]
        best_score_2 = list(best_values_2['score'])[0]

        clustering_2 = DBSCAN(eps=float(eps_val_2), min_samples=int(minsam_2)).fit(X_2)
        clusters_2 = data_phase_2.copy()
        clusters_2['DBScan_Clusters'] = clustering_2.labels_

        print(f"Second Iteration done for: {store_number}")

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

        clusters_2[['instore_transaction']] = Instore_Transactions_Scaler.inverse_transform(
            clusters_2[['instore_transaction_count_v2_scaled']]
        )
        clusters_2[['cust_store_dist']] = Distance_Scaler.inverse_transform(
            clusters_2[['cust_store_dist_scaled']]
        )
        clusters_2[['BOPIS_ORDER_counts']] = BOPIS_Scaler.inverse_transform(
            clusters_2[['BOPIS_Counts_scaled']]
        )

        # Combine clusters (exclude noise cluster 0)
        clusters = clusters[clusters['DBScan_Clusters'] != 0]
        clusters_2 = clusters_2[clusters_2['DBScan_Clusters'] != 0]

        clusters['GEOID'] = clusters[[
            'FIPS_STATE_CD', 'FIPS_COUNTY_CD', 'CENSUS_BLOCK_CD'
        ]].agg(''.join, axis=1)
        clusters_2['GEOID'] = clusters_2[[
            'FIPS_STATE_CD', 'FIPS_COUNTY_CD', 'CENSUS_BLOCK_CD'
        ]].agg(''.join, axis=1)

        clusters_final = pd.concat([clusters, clusters_2], ignore_index=True)
        clusters_final['Store Number'] = [store_number] * len(clusters_final)
        clusters_final = clusters_final[clusters_final['distance'] < config.MAX_DISTANCE]

        print(f"Cluster data prepared for: {store_number}")

        # Save cluster results
        output_file = config.CLUSTERS_OUTPUT_DIR / f"baby_{store_number}_clusters.csv"
        clusters_final[[
            'FIPS_STATE_CD', 'FIPS_COUNTY_CD', 'CENSUS_BLOCK_CD',
            'GEOID', 'Store Number'
        ]].drop_duplicates().to_csv(output_file, index=False)

        # Save parameters
        store_params = pd.DataFrame({
            'Phase_1_eps_val': [eps_val],
            'Phase_1_minsam': [minsam],
            'Phase_1_best_score': [best_score],
            'Phase_2_eps_val': [eps_val_2],
            'Phase_2_minsam': [minsam_2],
            'Phase_2_best_score': [best_score_2],
            'GEOID_1': [clusters['GEOID'].nunique()],
            'instore_transaction_1': [clusters['instore_transaction'].mean()],
            'cust_store_dist_1': [clusters['cust_store_dist'].mean()],
            'BOPIS_ORDER_counts_1': [clusters['BOPIS_ORDER_counts'].mean()],
            'GEOID': [clusters_final['GEOID'].nunique()],
            'instore_transaction': [clusters_final['instore_transaction'].mean()],
            'cust_store_dist': [clusters_final['cust_store_dist'].mean()],
            'BOPIS_ORDER_counts': [clusters_final['BOPIS_ORDER_counts'].mean()],
        })

        params_file = config.PARAMS_OUTPUT_DIR / f"baby_{store_number}_params.csv"
        store_params.to_csv(params_file, index=False)

        print(f"Clustering completed for: {store_number}")
        return 0

    except Exception as e:
        print(f"Error processing store {store_number}: {str(e)}")
        return 1


def main():
    """Main execution function"""
    print("=" * 60)
    print("BABY Trade Area Analysis - Starting")
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
        open_stores = bq_client.query(
            config.BABY_OPEN_STORES_QUERY,
            job_config=job_config
        ).result().to_dataframe()

        store_list = list(open_stores['BABY_OPEN_STORES'].unique())
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
