
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import haversine as hs
from haversine import Unit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud import bigquery_storage

#import data_import

#from data_import import store_list
#from data_import import Instore_Transaction_data_full
#from data_import import Distance_data_full
#from data_import import Home_Store_full


from multiprocessing import Process, Queue, Pool
import sys
import time

#list_df = []
#store_params = []

#print(store_list)

key_path = 'C:\\HALO Effect\\GCP_key.json'
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)
bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

job_config = bigquery.QueryJobConfig(allow_large_results=True)




#print("Creating Transaction Data")

Instore_Transaction_data_full = pd.read_pickle("C:\\HALO Effect\\BABY_Instore_Transactions_v2.pkl")



#print(Instore_Transaction_data_full.head())

#print("Transaction Data Created")

#print("Creating Distance Data")

Distance_data_full = pd.read_pickle("C:\\HALO Effect\\BABY_Distance_data_full.pkl")

#print(Distance_data_full.head())

#print("Distance Data Created")

#print("Creating Home Store Data")

Home_Store_full = pd.read_pickle("C:\\HALO Effect\\BABY_Home_Store_full.pkl")


BOPIS_full = pd.read_pickle("C:\\HALO Effect\\BABY_BOPIS_v2.pkl")



#print(Home_Store_full.head())

#print("Home Store Data Created")

#global store_list


QUERY = """SELECT distinct geo_id, pop_25_years_over FROM `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`; """

census_block_group = bq_client.query(QUERY, job_config=job_config).result().to_dataframe()
census_block_group['geo_id'] = census_block_group['geo_id'].apply(lambda x: str(x)[:-1])
census_block_group_v2 =  census_block_group.groupby(['geo_id'])['pop_25_years_over'].sum().reset_index()

census_block_group_v2['geo_id'].nunique()


def clustering_func(jobs2000):
     
    print("Clustering started for:"+str(jobs2000))
    
    #Filtering based on respective stores
    Instore_Transaction_data = Instore_Transaction_data_full[Instore_Transaction_data_full['store_nbr'] == int(jobs2000)]
    Distance_data = Distance_data_full[Distance_data_full['store_nbr'] == int(jobs2000)]
#    Home_Store = Home_Store_full[Home_Store_full['BABY_HOME_STORE_NBR'] == int(jobs2000)]    
    BOPIS = BOPIS_full[BOPIS_full['PICKUP_STORE'] == int(jobs2000)]
    
    Instore_Transaction_data['FIPS_STATE_CD'] = Instore_Transaction_data['FIPS_STATE_CD'].apply(lambda x: str(x).zfill(2) if len(str(x))!=2 else str(x))
    Instore_Transaction_data['FIPS_COUNTY_CD'] = Instore_Transaction_data['FIPS_COUNTY_CD'].apply(lambda x: str(x).zfill(3) if len(str(x))!=3 else str(x))
    Instore_Transaction_data['CENSUS_BLOCK_CD'] = Instore_Transaction_data['CENSUS_BLOCK_CD'].apply(lambda x: str(x).zfill(6) if len(str(x))!=6 else str(x))   
    Instore_Transaction_data['GEOID'] = Instore_Transaction_data[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)
    Instore_Transaction_data_v2 = Instore_Transaction_data.merge(census_block_group_v2, how='left',left_on='GEOID',right_on='geo_id')
    Instore_Transaction_data_v2['instore_transaction_count_v2'] = Instore_Transaction_data_v2['instore_transaction_count']/Instore_Transaction_data_v2['pop_25_years_over']
    
#    print(Instore_Transaction_data_v2.head(5))

    Distance_data['FIPS_STATE_CD'] = Distance_data['FIPS_STATE_CD'].apply(lambda x: str(x).zfill(2) if len(str(x))!=2 else str(x))
    Distance_data['FIPS_COUNTY_CD'] = Distance_data['FIPS_COUNTY_CD'].apply(lambda x: str(x).zfill(3) if len(str(x))!=3 else str(x))
    Distance_data['CENSUS_BLOCK_CD'] = Distance_data['CENSUS_BLOCK_CD'].apply(lambda x: str(x).zfill(6) if len(str(x))!=6 else str(x))
    Distance_data['GEOID'] = Distance_data[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)

#    print(Distance_data.head(5))    
    
#    Home_Store['FIPS_STATE_CD'] = Home_Store['FIPS_STATE_CD'].apply(lambda x: str(x).zfill(2) if len(str(x))!=2 else str(x))
#    Home_Store['FIPS_COUNTY_CD'] = Home_Store['FIPS_COUNTY_CD'].apply(lambda x: str(x).zfill(3) if len(str(x))!=3 else str(x))
#    Home_Store['CENSUS_BLOCK_CD'] = Home_Store['CENSUS_BLOCK_CD'].apply(lambda x: str(x).zfill(6) if len(str(x))!=6 else str(x))

    BOPIS['FIPS_STATE_CD'] = BOPIS['FIPS_STATE_CD'].apply(lambda x: str(x).zfill(2) if len(str(x))!=2 else str(x))
    BOPIS['FIPS_COUNTY_CD'] = BOPIS['FIPS_COUNTY_CD'].apply(lambda x: str(x).zfill(3) if len(str(x))!=3 else str(x))
    BOPIS['CENSUS_BLOCK_CD'] = BOPIS['CENSUS_BLOCK_CD'].apply(lambda x: str(x).zfill(6) if len(str(x))!=6 else str(x))
    BOPIS['GEOID'] = BOPIS[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)
    BOPIS_v2 = BOPIS.merge(census_block_group_v2, how='left',left_on='GEOID',right_on='geo_id')   
    BOPIS_v2['BOPIS_Counts'] =BOPIS_v2['CUSTOMER_ID']/BOPIS_v2['pop_25_years_over']
    
#    print(BOPIS_v2.head(5))    
    
    print("Initial transformation done for:"+str(jobs2000))
    
    
    #Merging data together
#    DD_HS = Distance_data.merge(Home_Store, how = 'outer', left_on=['FIPS_STATE_CD','FIPS_COUNTY_CD','CENSUS_BLOCK_CD',], right_on=['FIPS_STATE_CD','FIPS_COUNTY_CD','CENSUS_BLOCK_CD',])
    ITD_DD_HS = Instore_Transaction_data_v2.merge(Distance_data, how = 'outer', left_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD'], right_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD'])
    ITD_DD_HS_BP = ITD_DD_HS.merge(BOPIS_v2, how = 'outer', left_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD'], right_on=['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD'])
    
    ITD_DD_HS_BP.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    print("Merging data done for:"+str(jobs2000))    

    # Treating the Unknown values
    ITD_DD_HS_BP['instore_transaction_count_v2'] = ITD_DD_HS_BP['instore_transaction_count_v2'].fillna(0)
#    ITD_DD_HS_BP['Home_Store_counts'] = ITD_DD_HS_BP['Home_Store_counts'].fillna(0)
    ITD_DD_HS_BP['distance'] = ITD_DD_HS_BP['distance'].fillna(50)
    ITD_DD_HS_BP['distance_v2'] = 50 - ITD_DD_HS_BP['distance']
    ITD_DD_HS_BP['BOPIS_Counts'] = ITD_DD_HS_BP['BOPIS_Counts'].fillna(0)    

    print("Unknwon data treated for:"+str(jobs2000))     
    
    #Scaling values  
    ##BOPIS
    BOPIS_Scaler = StandardScaler()
    ITD_DD_HS_BP[['BOPIS_Counts_scaled']] = BOPIS_Scaler.fit_transform(ITD_DD_HS_BP[['BOPIS_Counts']])
#    print(1)
    ##Instore Transactions
    Instore_Transactions_Scaler = StandardScaler()
    ITD_DD_HS_BP[['instore_transaction_count_v2_scaled']] = Instore_Transactions_Scaler.fit_transform(ITD_DD_HS_BP[['instore_transaction_count_v2']])
#   p rint(2)
    ##Distance Data
    Distance_Scaler = StandardScaler()   
    ITD_DD_HS_BP[['cust_store_dist_scaled']] = Distance_Scaler.fit_transform(ITD_DD_HS_BP[['distance_v2']])
    ##HomeStore Data
#    Home_Store_Scaler = StandardScaler()
#    ITD_DD_HS_BP[['Home_Store_counts_scaled']] = Home_Store_Scaler.fit_transform(ITD_DD_HS_BP[['Home_Store_counts']])
    print("Scaled data for:"+str(jobs2000)) 
    
    
    #Passing the data into the model
#    X = ITD_DD_HS_BP.drop(['FIPS_STATE_CD','FIPS_COUNTY_CD','CENSUS_BLOCK_CD','EC_ORDER_NUM','instore_transaction_count','distance','Home_Store_counts'], axis = 1)
    X = ITD_DD_HS_BP[['BOPIS_Counts_scaled','instore_transaction_count_v2_scaled','cust_store_dist_scaled']]
#    print(X.head())
    
    # Getting the best parameters
    # Defining the list of hyperparameters to try
    eps_list=np.arange(start=0.1, stop=0.9, step=0.1)
    min_sample_list=np.arange(start=1, stop=10, step=1)

    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data=pd.DataFrame()   
    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:

            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial)

            if(len(np.unique(db.fit_predict(X)))>1):
                sil_score=silhouette_score(X, db.fit_predict(X))
            else:
                continue
            eps_parameter=str(eps_trial.round(1))
            min_sample_parameter = str(min_sample_trial)

            silhouette_scores_data=silhouette_scores_data.append(pd.DataFrame(data=[[sil_score,eps_parameter,min_sample_parameter]], columns=["score", "eps", "min_sample"]))

    # Finding out the best hyperparameters with highest Score
    best_values = silhouette_scores_data.sort_values(by='score', ascending=False).head(1)

    eps_val = list(best_values['eps'])[0]
    minsam = list(best_values['min_sample'])[0]
    best_score = list(best_values['score'])[0]

#   print(eps_val,minsam)

    clustering = DBSCAN(eps=float(eps_val), min_samples = int(minsam)).fit(X)
    clusters = ITD_DD_HS_BP
    clusters['DBScan_Clusters'] = clustering.labels_
    
    print("First Iteration done for:"+str(jobs2000))  
    
    data_phase_2 = clusters[clusters['DBScan_Clusters'] == 0]
    
#    X_2 = data_phase_2.drop(['FIPS_STATE_CD','FIPS_COUNTY_CD','CENSUS_BLOCK_CD','EC_ORDER_NUM','instore_transaction_count','distance','Home_Store_counts','DBScan_Clusters'], axis = 1)
    X_2 = data_phase_2[['BOPIS_Counts_scaled','instore_transaction_count_v2_scaled','cust_store_dist_scaled']]
    print(X_2.head())
    silhouette_scores_data_2=pd.DataFrame()
   
    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:

            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial)

            if(len(np.unique(db.fit_predict(X_2)))>1):
                sil_score=silhouette_score(X_2, db.fit_predict(X_2))
            else:
                continue
            eps_parameter=str(eps_trial.round(1))
            min_sample_parameter = str(min_sample_trial)

            silhouette_scores_data_2=silhouette_scores_data_2.append(pd.DataFrame(data=[[sil_score,eps_parameter,min_sample_parameter]], columns=["score", "eps", "min_sample"]))

    # Finding out the best hyperparameters with highest Score
    best_values_2 = silhouette_scores_data_2.sort_values(by='score', ascending=False).head(1)

    eps_val_2 = list(best_values_2['eps'])[0]
    minsam_2 = list(best_values_2['min_sample'])[0]
    best_score_2 = list(best_values_2['score'])[0]  


    clustering_2 = DBSCAN(eps=float(eps_val_2), min_samples = int(minsam_2)).fit(X_2)
    clusters_2 = data_phase_2
    clusters_2['DBScan_Clusters'] = clustering_2.labels_ 

    print("Second Iteration done for:"+str(jobs2000)) 

    clusters[['instore_transaction']] = Instore_Transactions_Scaler.inverse_transform(clusters[['instore_transaction_count_v2_scaled']])
    clusters[['cust_store_dist']] = Distance_Scaler.inverse_transform(clusters[['cust_store_dist_scaled']])
#    clusters[['Home_Store_counts']] = Home_Store_Scaler.inverse_transform(clusters[['Home_Store_counts_scaled']])
    clusters[['BOPIS_ORDER_counts']] = BOPIS_Scaler.inverse_transform(clusters[['BOPIS_Counts_scaled']])
  
    clusters_2[['instore_transaction']] = Instore_Transactions_Scaler.inverse_transform(clusters_2[['instore_transaction_count_v2_scaled']])
    clusters_2[['cust_store_dist']] = Distance_Scaler.inverse_transform(clusters_2[['cust_store_dist_scaled']])
#    clusters_2[['Home_Store_counts']] = Home_Store_Scaler.inverse_transform(clusters_2[['Home_Store_counts_scaled']])
    clusters_2[['BOPIS_ORDER_counts']] = BOPIS_Scaler.inverse_transform(clusters_2[['BOPIS_Counts_scaled']])

    clusters = clusters[clusters['DBScan_Clusters']!=0]  
    clusters_2 = clusters_2[clusters_2['DBScan_Clusters']!=0]
    
    clusters['GEOID'] = clusters[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)
    clusters_2['GEOID'] = clusters_2[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)

    clusters_final = pd.concat([clusters,clusters_2])  
    clusters_final['Store Number'] = [jobs2000]*int(len(clusters_final)) 

    clusters_final = clusters_final[clusters_final['distance'] < 50]

    print("Cluster data outputed for:"+str(jobs2000))     

    clusters_final[['FIPS_STATE_CD','FIPS_COUNTY_CD','CENSUS_BLOCK_CD','GEOID','Store Number']].drop_duplicates().to_csv("C:\\HALO Effect\\Baby_Results_BOPIS_Phase_2_no_HS\\"+str(jobs2000)+"_clusters.csv", index=False)

    store_params = pd.DataFrame()
    store_params['Phase_1_eps_val'] = [eps_val]
    store_params['Phase_1_minsam'] = [minsam]
    store_params['Phase_1_best_score'] = [best_score]
    store_params['Phase_2_eps_val'] = [eps_val_2]
    store_params['Phase_2_minsam'] = [minsam_2]
    store_params['Phase_2_best_score'] = [best_score_2]
    
    store_params['GEOID_1'] = clusters['GEOID'].nunique()
    store_params['instore_transaction_1'] = clusters['instore_transaction'].mean()
    store_params['cust_store_dist_1'] = clusters['cust_store_dist'].mean()
#    store_params['Home_Store_counts_1'] = clusters['Home_Store_counts']
    store_params['BOPIS_ORDER_counts_1'] = clusters['BOPIS_ORDER_counts'].mean()
    
    
    store_params['GEOID'] = clusters_final['GEOID'].nunique()
    store_params['instore_transaction'] = clusters_final['instore_transaction'].mean()
    store_params['cust_store_dist'] = clusters_final['cust_store_dist'].mean()
#    store_params['Home_Store_counts'] = clusters_final['Home_Store_counts']
    store_params['BOPIS_ORDER_counts'] = clusters_final['BOPIS_ORDER_counts'].mean()
    
    print("Best params outputed for:"+str(jobs2000)) 

    store_params.to_csv("C:\\HALO Effect\\Baby_Best_Params_BOPIS_Phase_2_no_HS\\"+str(jobs2000)+"_params.csv", index=False)
    
    print("Clustering completed for:"+str(jobs2000))
    
    return 0

    
      

if __name__ == "__main__": 
    print("Creating Store list")
    QUERY = """select distinct BABY_OPEN_STORES from `dw-bq-data-d00.QUANT_STG.SK_BABY_OPEN_STORES`;"""
    open_stores = bq_client.query(QUERY, job_config=job_config).result().to_dataframe()
#    print(open_stores.head())

    store_list = list(open_stores['BABY_OPEN_STORES'].unique()) 
    
    start = time.time()
    pool = Pool(50)
    results = pool.map_async(clustering_func, store_list)
    pool.close()
    pool.join()
    end = time.time()
    print("time spent: {}".format(end-start))  
    
