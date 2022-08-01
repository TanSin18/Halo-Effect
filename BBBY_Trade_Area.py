
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
#from sklearn.cluster import KMeans

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

Instore_Transaction_data_full = pd.read_pickle("C:\\HALO Effect\\BBBY_Open_Store_Transactions_v2.pkl")



#print(Instore_Transaction_data_full.head())

#print("Transaction Data Created")

#print("Creating Distance Data")

Distance_data_full = pd.read_pickle("C:\\HALO Effect\\BBBY_Open_Stores_Distance_data_full.pkl")

#print(Distance_data_full.head())

#print("Distance Data Created")

#print("Creating Home Store Data")

#Home_Store_full = pd.read_pickle("C:\\HALO Effect\\BABY_Home_Store_full.pkl")


BOPIS_full = pd.read_pickle("C:\\HALO Effect\\BBBY_open_store_BOPIS_v2.pkl")



#print(Home_Store_full.head())

#print("Home Store Data Created")

#global store_list


QUERY = """SELECT distinct geo_id, pop_25_years_over FROM `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`; """

census_block_group = bq_client.query(QUERY, job_config=job_config).result().to_dataframe()
census_block_group['geo_id'] = census_block_group['geo_id'].apply(lambda x: str(x)[:-1])
census_block_group_v2 =  census_block_group.groupby(['geo_id'])['pop_25_years_over'].sum().reset_index()

census_block_group_v2['geo_id'].nunique()


store_params = pd.read_csv("C:\\HALO Effect\\BBBY_best_params_stores.csv")

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
    X = ITD_DD_HS_BP[['instore_transaction_count_v2_scaled','cust_store_dist_scaled','BOPIS_Counts_scaled']]
#    print(X.head())
    
    store_params_v2 = store_params[store_params['Store_Number']==int(jobs2000)]
#   print(eps_val,minsam)

    clustering = DBSCAN(eps=float(store_params_v2['Phase_1_eps_val']), min_samples = int(store_params_v2['Phase_1_minsam'])).fit(X)
    clusters = ITD_DD_HS_BP
    clusters['DBScan_Clusters'] = clustering.labels_
    
    print("First Iteration done for:"+str(jobs2000))  
    

    clusters[['instore_transaction']] = Instore_Transactions_Scaler.inverse_transform(clusters[['instore_transaction_count_v2_scaled']])
    clusters[['cust_store_dist']] = Distance_Scaler.inverse_transform(clusters[['cust_store_dist_scaled']])
#    clusters[['Home_Store_counts']] = Home_Store_Scaler.inverse_transform(clusters[['Home_Store_counts_scaled']])
    clusters[['BOPIS_ORDER_counts']] = BOPIS_Scaler.inverse_transform(clusters[['BOPIS_Counts_scaled']])
  
    clusters = clusters[clusters['DBScan_Clusters']!=0]  
    
    clusters['GEOID'] = clusters[['FIPS_STATE_CD', 'FIPS_COUNTY_CD','CENSUS_BLOCK_CD']].agg(''.join, axis=1)

    clusters_final = clusters 
    clusters_final['Store Number'] = [jobs2000]*int(len(clusters_final)) 

    clusters_final = clusters_final[clusters_final['distance'] < 50]
    
    
    kmeans  = KMeans(n_clusters=2)

    kmeans_x = clusters_final[['distance']]
    kmeans.fit(kmeans_x)
    clusters_final['kmeans_cluster'] = kmeans.labels_
   
    clus_num = clusters_final.groupby(['kmeans_cluster'])['distance'].mean().reset_index().sort_values(['distance'], ascending=True).head(1)['kmeans_cluster']
   
    clusters_final_v2 = clusters_final[clusters_final['kmeans_cluster'].isin(list(clus_num))]

      

    clusters_final_v2[['FIPS_STATE_CD','FIPS_COUNTY_CD','CENSUS_BLOCK_CD','GEOID','Store Number']].drop_duplicates().to_csv("C:\\HALO Effect\\BBBY_Results_BOPIS_Phase_1_no_HS\\"+str(jobs2000)+"_clusters.csv", index=False)

    print("Cluster data outputed for:"+str(jobs2000)) 
    
#    store_params.to_csv("C:\\HALO Effect\\BBBY_Best_Params_BOPIS_Phase_2_no_HS\\"+str(jobs2000)+"_params.csv", index=False)
    
    print("Clustering completed for:"+str(jobs2000))
    
    return 0

    
      

if __name__ == "__main__": 
    print("Creating Store list")
    QUERY = """select distinct BBBY__OPEN_STORES from `dw-bq-data-d00.QUANT_STG.SK_OPEN_STORES_JULY_2021`;"""
    open_stores = bq_client.query(QUERY, job_config=job_config).result().to_dataframe()
#    print(open_stores.head())

    store_list = list(open_stores['BBBY__OPEN_STORES'].unique()) 
    print(len(store_list))
    
    start = time.time()
    pool = Pool(60)
    results = pool.map_async(clustering_func,store_list)
    pool.close()
    pool.join()
    end = time.time()
    print("time spent: {}".format(end-start))  
    
