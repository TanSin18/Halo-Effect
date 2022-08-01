# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:36:03 2021

@author: SK0759
"""

import pandas as pd
import numpy as np
from dateutil.easter import easter
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import random
from fbprophet.diagnostics import cross_validation
import warnings
warnings.filterwarnings('ignore')
from fbprophet.diagnostics import performance_metrics
from sklearn import metrics







def forecast_model_aug(df):
    #sanity check for column names
    print(df.columns)
    
    #only using data till Sept for training 
#    df_july = df[df['ds'] <= '2020-07-31']
    
#     additive decompostion model:
#     y(t) = trend + seasonality + holiday_effect + residual_error
    
#     multiplicative decomp..
#     y(t) = trend * seasonality * holiday_effect * residual_error
    
    #fitting the model
    m =Prophet(seasonality_mode='multiplicative',changepoint_prior_scale=0.001)
#     m.add_country_holidays(country_name='US')
    m.fit(df)
   
        
    #forecasting for the next 1 months
    future = m.make_future_dataframe(periods=18, freq='MS')
    forecast = m.predict(future)
    
    #checking the trends and other model components
    #print(m.plot(forecast))
    #print(m.plot_components(forecast))
    
    results_aug = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(18)
    results_aug['actual'] = df[-18:]['y']
    results_aug.columns = ['Date', 'Predicted_gross_Sales', 'Predicted_Lower', 'Predicted_Upper', 'Actual Counts']
    print(results_aug.columns)
    
    return results_aug


# =============================================================================
# Open Stores
# =============================================================================
open_df = pd.read_csv("C:\\Users\\sk0759\\Desktop\\Halo Effect\\open_Stores_phase1_17_18_19_20_21.csv")
open_df['months'] = open_df['months'].apply(lambda x:str(x).zfill(2))



store_number=[]
MAPE_score=[]
predictions=[]

for store in list(open_df['store_number'].unique()):
    print(store)
    open_df_1 = open_df[(open_df['store_number']==store)]
    open_df_1['DATE'] = open_df_1.apply(lambda x:str(x['yearss'])+ '-' + str(x['months'])+ '-' + '01',axis=1)

    df_train = open_df_1[(open_df_1['yearss']!=2021) & ((open_df_1['yearss']!=2020))]
    
    df_test = open_df_1[(open_df_1['yearss']==2021) | (open_df_1['yearss']==2020)]
    
    
    df_train = df_train[['DATE', 'f0_']]
    
    df_train = df_train.rename(columns = {'DATE':'ds','f0_':'y'})

    df_train_results = forecast_model_aug(df_train)
    
    df_train_results.dtypes
    df_test.dtypes
    df_test['DATE'] = df_test['DATE'].astype('datetime64[ns]')
    
    
    data_merged = df_train_results.merge(df_test, how='inner',left_on='Date', right_on='DATE')
    
    predictions.append(data_merged)
       
    store_number.append(store)
    MAPE_score.append(metrics.mean_absolute_error(data_merged['f0_'],data_merged['Predicted_gross_Sales'])/np.mean(data_merged['f0_'])*100)
#    print(metrics.mean_absolute_error(data_merged['f0_'],data_merged['Predicted_gross_Sales'])/np.mean(data_merged['f0_'])*100)

open_result_data = pd.concat(predictions)

open_result_data.rename(columns={'f0_':'Actual_gross_Sales'},inplace=True)

open_result_data['Pct_Difference'] = ((open_result_data['Actual_gross_Sales']/open_result_data['Predicted_gross_Sales'])-1)

# =============================================================================
# Store CLusters
# =============================================================================


store_clusters = pd.read_excel('C:\\Users\\sk0759\\Desktop\\Halo Effect\\Store_Clusters.xlsx',sheet_name='Store_clusters2019Fiscal')

store_clusters.fillna('UNKNOWN', inplace=True)


closed_store_info = closed_Store_factor.merge(store_clusters, how='left', left_on=['store_number'], right_on=['Store #'])[['store_number','Clusters','closure_factor','Region']]
                                              
                                       
                            
# =============================================================================
# mapping closed and open stores
# =============================================================================

analysis = closed_store_info.merge(store_clusters, how='left', left_on=['Clusters','Region'], right_on=['Clusters','Region'])[['store_number','Clusters','Region','closure_factor','Store #']]
                                   
analysis_v2 = analysis[~(analysis['store_number'] == analysis['Store #'])].rename(columns={'Store #':'open_store_number'})
      

analysis_v2.rename(columns={'store_number': 'closed_store_number','closure_factor': 'closed_store_closure_factor'},inplace=True)

analysis_v2['closed_store_number'].nunique()

analysis_v5 = analysis_v2.merge(open_Store_factor, how='inner', left_on=['open_store_number'], right_on = ['store_number'])[['closed_store_number','closed_store_closure_factor','Region','open_store_number','closure_factor']]

analysis_v5['closed_store_number'].nunique()

analysis_v6 = analysis_v5.groupby(['closed_store_number','closed_store_closure_factor','Region']).agg({'closure_factor':'mean','open_store_number':'nunique'}).reset_index().rename(columns={'closure_factor':'avg_opening_closure_factor','open_store_number':'Avg # open stores'})

#['closure_factor'].mean().reset_index().rename(columns={'closure_factor':'avg_opening_closure_factor'})

analysis_v5.to_excel('C:\\Users\\sk0759\\Desktop\\Halo Effect\\closed_open_mapping.xlsx', index=False)


#analysis_v6.to_csv('C:\\Users\\sk0759\\Desktop\\Halo Effect\\calibrated_closure_factor_v7.csv', index=False)



# =============================================================================
# similar open stores
# =============================================================================

mappings = pd.read_excel("C:\\Users\\sk0759\\Desktop\\Halo Effect\\closed_open_mapping.xlsx")

mappings['closed_store_number'].nunique()

mappings.columns

mappings.head()

merged_Data = open_result_data.merge(mappings, how='inner', left_on='store_number', right_on='open_store_number').reset_index()


merged_Data_analysis = merged_Data.groupby(['closed_store_number','DATE']).agg({'Actual_gross_Sales':'mean','Predicted_gross_Sales':'mean','Pct_Difference':'mean'}).reset_index()


merged_Data_analysis.to_csv("C:\\Users\\sk0759\\Desktop\\Halo Effect\\covid_factor_calculation_phase1.csv", index=False)

# =============================================================================
# closed stores
# =============================================================================

df = pd.read_csv("C:\\Users\\sk0759\\Desktop\\Halo Effect\\closed_Stores_phase1_17_18_19_20_21.csv")
df['months'] = df['months'].apply(lambda x:str(x).zfill(2))


closed_store_number=[]
closed_MAPE_score=[]
closed_predictions=[]

for store in list(df['store_number'].unique()):
    print(store)
    df_1 = df[(df['store_number']==store)]
    df_1['DATE'] = df_1.apply(lambda x:str(x['yearss'])+ '-' + str(x['months'])+ '-' + '01',axis=1)

 
    df_train = df_1[(df_1['yearss']!=2021) & ((df_1['yearss']!=2020))]
    
    df_test = df_1[(df_1['yearss']==2021) | (df_1['yearss']==2020)]
    
    
    df_train = df_train[['DATE', 'f0_']]
    
    df_train = df_train.rename(columns = {'DATE':'ds','f0_':'y'})

    df_train_results = forecast_model_aug(df_train)
    
    df_train_results.dtypes
    df_test.dtypes
    df_test['DATE'] = df_test['DATE'].astype('datetime64[ns]')
    
    
    data_merged = df_train_results.merge(df_test, how='inner',left_on='Date', right_on='DATE')
    
    closed_predictions.append(data_merged)
       
    closed_store_number.append(store)
    closed_MAPE_score.append(metrics.mean_absolute_error(data_merged['f0_'],data_merged['Predicted_gross_Sales'])/np.mean(data_merged['f0_'])*100)
#    print(metrics.mean_absolute_error(data_merged['f0_'],data_merged['Predicted_gross_Sales'])/np.mean(data_merged['f0_'])*100)

closed_result_data = pd.concat(closed_predictions)

closed_result_data.rename(columns={'f0_':'Actual_gross_Sales'},inplace=True)

closed_result_data.to_csv("C:\\Users\\sk0759\\Desktop\\Halo Effect\\closed_stores_phase1_predictions.csv", index=False) 
