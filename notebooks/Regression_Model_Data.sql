=====================================================================================================================================================================================================
--Creating store customer list (closed stores BBBY)
=====================================================================================================================================================================================================
drop table store_customer_list if exists;
create table store_customer_list as
select distinct STORE_NUMBER, CB.FIPS_STATE_CD, CB.FIPS_COUNTY_CD, CB.CENSUS_BLOCK_CD, CUR.customer_id, CUR.MAILING_ADDRESS_ID, CUR.ADDRESS_ID
from EDW_MCF_VW..MAILING_ADDRESS MA
join (select 
distinct
STORE_NUMBER, 
cast(lpad(FIPS_STATE_CD,2,0) as varchar(10)) FIPS_STATE_CD,
cast(lpad(FIPS_COUNTY_CD,3,0) as varchar(10)) FIPS_COUNTY_CD,
cast(lpad(CENSUS_BLOCK_CD,6,0) as varchar(10)) CENSUS_BLOCK_CD
--from store_clusters
from Closure_store_clusters_BOPIS_Phase_1_no_HS -- Trade Area to store mapping
) CB
on MA.FIPS_STATE_CD = CB.FIPS_STATE_CD
and MA.FIPS_COUNTY_CD = CB.FIPS_COUNTY_CD
and MA.CENSUS_BLOCK_CD = CB.CENSUS_BLOCK_CD
join EDW_MCF_VW..CUSTOMER_CURR CUR
on MA.MAILING_ADDRESS_ID = CUR.MAILING_ADDRESS_ID;
=====================================================================================================================================================================================================
--Creating store customer list (open stores BBBY)
=====================================================================================================================================================================================================
drop table store_customer_list if exists;
create table store_customer_list as
select distinct STORE_NUMBER, CB.FIPS_STATE_CD, CB.FIPS_COUNTY_CD, CB.CENSUS_BLOCK_CD, CUR.customer_id, CUR.MAILING_ADDRESS_ID, CUR.ADDRESS_ID
from EDW_MCF_VW..MAILING_ADDRESS MA
join (select 
distinct
STORE_NUMBER, 
cast(lpad(FIPS_STATE_CD,2,0) as varchar(10)) FIPS_STATE_CD,
cast(lpad(FIPS_COUNTY_CD,3,0) as varchar(10)) FIPS_COUNTY_CD,
cast(lpad(CENSUS_BLOCK_CD,6,0) as varchar(10)) CENSUS_BLOCK_CD
--from store_clusters
from BBBBY_store_clusters_BOPIS_Phase_1_no_HS -- Trade Area to store mapping
) CB
on MA.FIPS_STATE_CD = CB.FIPS_STATE_CD
and MA.FIPS_COUNTY_CD = CB.FIPS_COUNTY_CD
and MA.CENSUS_BLOCK_CD = CB.CENSUS_BLOCK_CD
join EDW_MCF_VW..CUSTOMER_CURR CUR
on MA.MAILING_ADDRESS_ID = CUR.MAILING_ADDRESS_ID;
=====================================================================================================================================================================================================
--Creating 2019 customer transaction (BBBY)
=====================================================================================================================================================================================================
drop table SK_CUSTOMER_5_YEARS_2019 if exists;
create table SK_CUSTOMER_5_YEARS_2019 as
select 
distinct
XXX.*,
YYY.DMA_CD store_DMA,
ZZZ.DMA_NAME store_DMA_Name,
CUSTOMER_LOCS.FIPS_STATE_CD,
CUSTOMER_LOCS.FIPS_COUNTY_CD,
CUSTOMER_LOCS.POSTAL_CD,
CUSTOMER_LOCS.CENSUS_BLOCK_CD,
CUSTOMER_LOCS.LATITUDE_MEAS cust_Latitude,
CUSTOMER_LOCS.LONGITUDE_MEAS cust_Longitude,
YYY.store_nbr,
LOCS.LATITUDE_MEAS store_Latitude,
LOCS.LONGITUDE_MEAS store_Longitude


from

(
select 
distinct sdm.TRANS_ID,
sdm.ORDER_TRANS_ID,
xref.CUSTOMER_ID,
cc.address_id,
cc.MAILING_ADDRESS_ID,
cc.household_id,
TRANS_LOCATION_ID,
sdm.TRANS_BOOKED_DT,
sdm.sales, 
sdm.gross_Sales, 
sdm.ECOM_ORDER_SOURCE_CD,
case when sdm.ECOM_ORDER_SOURCE_CD in ('U', 'BBS') then 'instore' else 'online' end as instore_online_ind,
extract(year from TRANS_BOOKED_DT) yearss,
extract(week from TRANS_BOOKED_DT) weeks,
extract(month from TRANS_BOOKED_DT) monthss

	  from SALES_DATAMART_VW.ADMIN.SDM_SALES_TXN_SUM_2019 sdm -- 2021
	  inner join EDW_MCF_VW..CUSTOMER_TXN_XREF xref
	  on sdm.TRANS_ID = xref.TRANS_ID
	  inner join EDW_MCF_VW..CUSTOMER_CURR cc 
	  on xref.CUSTOMER_ID = cc.CUSTOMER_ID
	  where sdm.CONCEPT_FORMAT_ID = 1 
		and SALES_TRANS_TYPE_CD IN ('S','M')
		and xref.CUSTOMER_ID > 0
		AND cc.GHOST_RECORD_IND = 'N'
	    AND cc.CUSTOMER_PURGE_IND = 'N'
		and BOPUS_LN_IND != 'Y'

) XXX
join (select distinct LOCATION_ID, STORE_NBR, DMA_CD from EDW_MCF_VW..LOCATION where LOCATION_ID is not null ) YYY
on XXX.TRANS_LOCATION_ID = YYY.LOCATION_ID
join (select distinct DMA_CD, DMA_NAME from EDW_MCF_VW..DIRECT_MARKETING_AREA where DMA_CD is not null) ZZZ
on YYY.DMA_CD = ZZZ.DMA_CD
join (select LOCATION_ID, LATITUDE_MEAS, LONGITUDE_MEAS from EDW_MCF_VW..LT_LOCATION_GEO_COORD where LOCATION_ID is not null) LOCS
on YYY.LOCATION_ID = LOCS.LOCATION_ID
join (select MAILING_ADDRESS_ID, FIPS_STATE_CD, FIPS_COUNTY_CD, POSTAL_CD, CENSUS_BLOCK_CD, LATITUDE_MEAS, LONGITUDE_MEAS from EDW_MCF_VW..MAILING_ADDRESS where MAILING_ADDRESS_ID is not null) CUSTOMER_LOCS
on XXX.MAILING_ADDRESS_ID = CUSTOMER_LOCS.MAILING_ADDRESS_ID
;
=====================================================================================================================================================================================================
--weekly level info for closed stores
=====================================================================================================================================================================================================
select 
online_sales.*,
SEASONALITY.CALENDAR_SEASON_CD,
instore_sales.instore_customer_counts,
instore_sales.instore_transaction_counts,
instore_sales.instore_gross_sales,

NEAR_indicator.ALL_NEAR_N,
NEAR_indicator.ALL_NEAR_E,
NEAR_indicator.ALL_NEAR_A,
NEAR_indicator.ALL_NEAR_R,

NEAR_indicator.INSTORE_NEAR_N,
NEAR_indicator.INSTORE_NEAR_E,
NEAR_indicator.INSTORE_NEAR_A,
NEAR_indicator.INSTORE_NEAR_R,

NEAR_indicator.ONLINE_NEAR_N,
NEAR_indicator.ONLINE_NEAR_E,
NEAR_indicator.ONLINE_NEAR_A,
NEAR_indicator.ONLINE_NEAR_R,

STORE_INFORMATION.census_block_counts,
STORE_INFORMATION.STORE_VOLUME_CD,
STORE_INFORMATION.FASHION_ATTRIBUTE_CD,
STORE_INFORMATION.opened_days


from
(
select 
STORE_NUMBER,
census_block_counts,
STORE_VOLUME_CD,
FASHION_ATTRIBUTE_CD,
opened_days
from
(
select 
STORE_NUMBER,
count(FIPS_STATE_CD || FIPS_COUNTY_CD ||CENSUS_BLOCK_CD) census_block_counts
from
(
select 
distinct
STORE_NUMBER, 
cast(lpad(FIPS_STATE_CD,2,0) as varchar(10)) FIPS_STATE_CD,
cast(lpad(FIPS_COUNTY_CD,3,0) as varchar(10)) FIPS_COUNTY_CD,
cast(lpad(CENSUS_BLOCK_CD,6,0) as varchar(10)) CENSUS_BLOCK_CD
from Closure_store_clusters_BOPIS_Phase_2_no_HS -- has to be changed for open BBBY stores
) AA
group by 1
order by 1
) CENSUS_INFO
join
(
select 
STORE_NBR, 
STORE_VOLUME_CD,
extract(days from (now() - LOCATION_OPEN_DT)) opened_days ,
FASHION_ATTRIBUTE_CD
from EDW_MCF_VW.ADMIN.LOCATION 
) STORE_INFO
on CENSUS_INFO.STORE_NUMBER = STORE_INFO.STORE_NBR
) STORE_INFORMATION

join
--customer NEAR indicator
(
select 
weeks,
STORE_NUMBER,
sum(ALL_NEAR_N) ALL_NEAR_N,
sum(ALL_NEAR_E) ALL_NEAR_E,
sum(ALL_NEAR_A) ALL_NEAR_A,
sum(ALL_NEAR_R) ALL_NEAR_R,

sum(INSTORE_NEAR_N) INSTORE_NEAR_N,
sum(INSTORE_NEAR_E) INSTORE_NEAR_E,
sum(INSTORE_NEAR_A) INSTORE_NEAR_A,
sum(INSTORE_NEAR_R) INSTORE_NEAR_R,

sum(ONLINE_NEAR_N) ONLINE_NEAR_N,
sum(ONLINE_NEAR_E) ONLINE_NEAR_E,
sum(ONLINE_NEAR_A) ONLINE_NEAR_A,
sum(ONLINE_NEAR_R) ONLINE_NEAR_R

from
(
select 
distinct 
customer_id,
STORE_NUMBER,
extract(week from TRANS_BOOKED_DT) weeks,
ALL_NEAR_N,
ALL_NEAR_E,
ALL_NEAR_A,
ALL_NEAR_R,

INSTORE_NEAR_N,
INSTORE_NEAR_E,
INSTORE_NEAR_A,
INSTORE_NEAR_R,

ONLINE_NEAR_N,
ONLINE_NEAR_E,
ONLINE_NEAR_A,
ONLINE_NEAR_R

from
(
select distinct customer_id, STORE_NUMBER,BBB_ALL_NEAR, BBB_INSTORE_NEAR, BBB_ONLINE_NEAR, TRANS_BOOKED_DT,
case when BBB_ALL_NEAR = 'N' then 1 else 0 end as ALL_NEAR_N,
case when BBB_ALL_NEAR = 'E' then 1 else 0 end as ALL_NEAR_E,
case when BBB_ALL_NEAR = 'A' then 1 else 0 end as ALL_NEAR_A,
case when BBB_ALL_NEAR = 'R' then 1 else 0 end as ALL_NEAR_R,

case when BBB_INSTORE_NEAR = 'N' then 1 else 0 end as INSTORE_NEAR_N,
case when BBB_INSTORE_NEAR = 'E' then 1 else 0 end as INSTORE_NEAR_E,
case when BBB_INSTORE_NEAR = 'A' then 1 else 0 end as INSTORE_NEAR_A,
case when BBB_INSTORE_NEAR = 'R' then 1 else 0 end as INSTORE_NEAR_R,

case when BBB_ONLINE_NEAR = 'N' then 1 else 0 end as ONLINE_NEAR_N,
case when BBB_ONLINE_NEAR = 'E' then 1 else 0 end as ONLINE_NEAR_E,
case when BBB_ONLINE_NEAR = 'A' then 1 else 0 end as ONLINE_NEAR_A,
case when BBB_ONLINE_NEAR = 'R' then 1 else 0 end as ONLINE_NEAR_R



from
(

select 
AA.*,
BB.TRANS_ID, BB.TRANS_BOOKED_DT, BB.BBB_ALL_NEAR, BB.BBB_INSTORE_NEAR, BB.BBB_ONLINE_NEAR
from
(select * from store_customer_list) AA
join (select distinct customer_id,TRANS_ID, TRANS_BOOKED_DT, BBB_ALL_NEAR, BBB_INSTORE_NEAR, BBB_ONLINE_NEAR
from analytics_vw..ca_cust_near where TRANS_BOOKED_DT between '2019-01-01 00:00:00' and '2019-12-31 00:00:00' and CONCEPT_FORMAT_ID = 1) BB
on AA.customer_id = BB.customer_id
) xx
) YY
) ZZ
group by 1,2
order by 1,2
) NEAR_indicator
on STORE_INFORMATION.STORE_NUMBER = NEAR_indicator.STORE_NUMBER

join

--customer online sales

(
select 
STORE_NUMBER,
extract(week from TRANS_BOOKED_DT) weeks,
count(distinct XX.customer_id) online_customer_counts,
count(distinct order_trans_id) online_transaction_counts,
sum(gross_sales) online_gross_sales

from
(
select distinct customer_id, order_trans_id, trans_booked_dt, gross_sales
from SK_CUSTOMER_5_YEARS_2019
where INSTORE_ONLINE_IND = 'online'
) xx
join
store_customer_list YY
on XX.customer_id = YY.customer_id
group by 1,2
order by 1,2
) online_sales

on NEAR_indicator.STORE_NUMBER = online_sales.STORE_NUMBER
and NEAR_indicator.weeks = online_sales.weeks

--customer instore sales
join
(
select 
STORE_NUMBER,
extract(week from TRANS_BOOKED_DT) weeks,
count(distinct XX.customer_id) instore_customer_counts,
count(distinct order_trans_id) instore_transaction_counts,
sum(gross_sales) instore_gross_sales

from
(
select distinct customer_id, order_trans_id, trans_booked_dt, gross_sales
from SK_CUSTOMER_5_YEARS_2019
where INSTORE_ONLINE_IND = 'instore'
) xx
join
store_customer_list YY
on XX.customer_id = YY.customer_id
group by 1,2
order by 1,2
) instore_sales

on instore_sales.STORE_NUMBER = online_sales.STORE_NUMBER
and instore_sales.weeks = online_sales.weeks

join
(
select 
distinct weeks, CALENDAR_SEASON_CD
from
(
select
distinct
extract(week from CALENDAR_DT) as weeks,
CALENDAR_SEASON_CD,
row_number() over (partition by weeks order by weeks) as rn 
from EDW_MCF_VW..CALENDAR_DATE 
--where CALENDAR_DT between '2019-01-01 00:00:00' and '2019-12-31 00:00:00'
order by 1
) xx
where rn=1
order by 1
) SEASONALITY

on instore_sales.weeks = SEASONALITY.weeks
order by 1,2;
=====================================================================================================================================================================================================
--Creating store customer list (BABY stores)
=====================================================================================================================================================================================================
drop table BABY_store_customer_list if exists;
create table BABY_store_customer_list as
select distinct STORE_NUMBER, CB.FIPS_STATE_CD, CB.FIPS_COUNTY_CD, CB.CENSUS_BLOCK_CD, CUR.customer_id, CUR.MAILING_ADDRESS_ID, CUR.ADDRESS_ID
from EDW_MCF_VW..MAILING_ADDRESS MA
join (select 
distinct
STORE_NUMBER, 
cast(lpad(FIPS_STATE_CD,2,0) as varchar(10)) FIPS_STATE_CD,
cast(lpad(FIPS_COUNTY_CD,3,0) as varchar(10)) FIPS_COUNTY_CD,
cast(lpad(CENSUS_BLOCK_CD,6,0) as varchar(10)) CENSUS_BLOCK_CD
--from store_clusters
from SK_BABY_STORE_CLUSTERS
) CB
on MA.FIPS_STATE_CD = CB.FIPS_STATE_CD
and MA.FIPS_COUNTY_CD = CB.FIPS_COUNTY_CD
and MA.CENSUS_BLOCK_CD = CB.CENSUS_BLOCK_CD
join EDW_MCF_VW..CUSTOMER_CURR CUR
on MA.MAILING_ADDRESS_ID = CUR.MAILING_ADDRESS_ID;
=====================================================================================================================================================================================================
--Creating 2019 customer transaction (BABY)
=====================================================================================================================================================================================================
drop table SK_CUSTOMER_5_YEARS_2019_BABY if exists;
create table SK_CUSTOMER_5_YEARS_2019_BABY as
select 
distinct
XXX.*,
YYY.DMA_CD store_DMA,
ZZZ.DMA_NAME store_DMA_Name,
CUSTOMER_LOCS.FIPS_STATE_CD,
CUSTOMER_LOCS.FIPS_COUNTY_CD,
CUSTOMER_LOCS.POSTAL_CD,
CUSTOMER_LOCS.CENSUS_BLOCK_CD,
CUSTOMER_LOCS.LATITUDE_MEAS cust_Latitude,
CUSTOMER_LOCS.LONGITUDE_MEAS cust_Longitude,
YYY.store_nbr,
LOCS.LATITUDE_MEAS store_Latitude,
LOCS.LONGITUDE_MEAS store_Longitude


from

(
select 
distinct sdm.TRANS_ID,
sdm.ORDER_TRANS_ID,
xref.CUSTOMER_ID,
cc.address_id,
cc.MAILING_ADDRESS_ID,
cc.household_id,
TRANS_LOCATION_ID,
sdm.TRANS_BOOKED_DT,
sdm.sales, 
sdm.gross_Sales, 
sdm.ECOM_ORDER_SOURCE_CD,
case when sdm.ECOM_ORDER_SOURCE_CD in ('U', 'BAS') then 'instore' else 'online' end as instore_online_ind,
extract(year from TRANS_BOOKED_DT) yearss,
extract(week from TRANS_BOOKED_DT) weeks,
extract(month from TRANS_BOOKED_DT) monthss

	  from SALES_DATAMART_VW.ADMIN.SDM_SALES_TXN_SUM_2019 sdm -- 2021
	  inner join EDW_MCF_VW..CUSTOMER_TXN_XREF xref
	  on sdm.TRANS_ID = xref.TRANS_ID
	  inner join EDW_MCF_VW..CUSTOMER_CURR cc 
	  on xref.CUSTOMER_ID = cc.CUSTOMER_ID
	  where sdm.CONCEPT_FORMAT_ID = 3 
		and SALES_TRANS_TYPE_CD IN ('S','M')
		and xref.CUSTOMER_ID > 0
		AND cc.GHOST_RECORD_IND = 'N'
	    AND cc.CUSTOMER_PURGE_IND = 'N'

) XXX
join (select distinct LOCATION_ID, STORE_NBR, DMA_CD from EDW_MCF_VW..LOCATION where LOCATION_ID is not null ) YYY
on XXX.TRANS_LOCATION_ID = YYY.LOCATION_ID
join (select distinct DMA_CD, DMA_NAME from EDW_MCF_VW..DIRECT_MARKETING_AREA where DMA_CD is not null) ZZZ
on YYY.DMA_CD = ZZZ.DMA_CD
join (select LOCATION_ID, LATITUDE_MEAS, LONGITUDE_MEAS from EDW_MCF_VW..LT_LOCATION_GEO_COORD where LOCATION_ID is not null) LOCS
on YYY.LOCATION_ID = LOCS.LOCATION_ID
join (select MAILING_ADDRESS_ID, FIPS_STATE_CD, FIPS_COUNTY_CD, POSTAL_CD, CENSUS_BLOCK_CD, LATITUDE_MEAS, LONGITUDE_MEAS from EDW_MCF_VW..MAILING_ADDRESS where MAILING_ADDRESS_ID is not null) CUSTOMER_LOCS
on XXX.MAILING_ADDRESS_ID = CUSTOMER_LOCS.MAILING_ADDRESS_ID
;
=====================================================================================================================================================================================================
--weekly level info for BABY open stores
=====================================================================================================================================================================================================
select 
online_sales.*,
SEASONALITY.CALENDAR_SEASON_CD,
instore_sales.instore_customer_counts,
instore_sales.instore_transaction_counts,
instore_sales.instore_gross_sales,

NEAR_indicator.ALL_NEAR_N,
NEAR_indicator.ALL_NEAR_E,
NEAR_indicator.ALL_NEAR_A,
NEAR_indicator.ALL_NEAR_R,

NEAR_indicator.INSTORE_NEAR_N,
NEAR_indicator.INSTORE_NEAR_E,
NEAR_indicator.INSTORE_NEAR_A,
NEAR_indicator.INSTORE_NEAR_R,

NEAR_indicator.ONLINE_NEAR_N,
NEAR_indicator.ONLINE_NEAR_E,
NEAR_indicator.ONLINE_NEAR_A,
NEAR_indicator.ONLINE_NEAR_R,

STORE_INFORMATION.census_block_counts,
STORE_INFORMATION.STORE_VOLUME_CD,
STORE_INFORMATION.FASHION_ATTRIBUTE_CD,
STORE_INFORMATION.opened_days


from
(
select 
STORE_NUMBER,
census_block_counts,
STORE_VOLUME_CD,
FASHION_ATTRIBUTE_CD,
opened_days
from
(
select 
STORE_NUMBER,
count(FIPS_STATE_CD || FIPS_COUNTY_CD ||CENSUS_BLOCK_CD) census_block_counts
from
(
select 
distinct
STORE_NUMBER, 
cast(lpad(FIPS_STATE_CD,2,0) as varchar(10)) FIPS_STATE_CD,
cast(lpad(FIPS_COUNTY_CD,3,0) as varchar(10)) FIPS_COUNTY_CD,
cast(lpad(CENSUS_BLOCK_CD,6,0) as varchar(10)) CENSUS_BLOCK_CD
from sk_baby_store_clusters
) AA
group by 1
order by 1
) CENSUS_INFO
join
(
select 
STORE_NBR, 
STORE_VOLUME_CD,
extract(days from (now() - LOCATION_OPEN_DT)) opened_days ,
FASHION_ATTRIBUTE_CD
from EDW_MCF_VW.ADMIN.LOCATION 
) STORE_INFO
on CENSUS_INFO.STORE_NUMBER = STORE_INFO.STORE_NBR
) STORE_INFORMATION

join
--customer NEAR indicator
(
select 
weeks,
STORE_NUMBER,
sum(ALL_NEAR_N) ALL_NEAR_N,
sum(ALL_NEAR_E) ALL_NEAR_E,
sum(ALL_NEAR_A) ALL_NEAR_A,
sum(ALL_NEAR_R) ALL_NEAR_R,

sum(INSTORE_NEAR_N) INSTORE_NEAR_N,
sum(INSTORE_NEAR_E) INSTORE_NEAR_E,
sum(INSTORE_NEAR_A) INSTORE_NEAR_A,
sum(INSTORE_NEAR_R) INSTORE_NEAR_R,

sum(ONLINE_NEAR_N) ONLINE_NEAR_N,
sum(ONLINE_NEAR_E) ONLINE_NEAR_E,
sum(ONLINE_NEAR_A) ONLINE_NEAR_A,
sum(ONLINE_NEAR_R) ONLINE_NEAR_R

from
(
select 
distinct 
customer_id,
STORE_NUMBER,
extract(week from TRANS_BOOKED_DT) weeks,
ALL_NEAR_N,
ALL_NEAR_E,
ALL_NEAR_A,
ALL_NEAR_R,

INSTORE_NEAR_N,
INSTORE_NEAR_E,
INSTORE_NEAR_A,
INSTORE_NEAR_R,

ONLINE_NEAR_N,
ONLINE_NEAR_E,
ONLINE_NEAR_A,
ONLINE_NEAR_R

from
(
select distinct customer_id, STORE_NUMBER,BAB_ALL_NEAR, BAB_INSTORE_NEAR, BAB_ONLINE_NEAR, TRANS_BOOKED_DT,
case when BAB_ALL_NEAR = 'N' then 1 else 0 end as ALL_NEAR_N,
case when BAB_ALL_NEAR = 'E' then 1 else 0 end as ALL_NEAR_E,
case when BAB_ALL_NEAR = 'A' then 1 else 0 end as ALL_NEAR_A,
case when BAB_ALL_NEAR = 'R' then 1 else 0 end as ALL_NEAR_R,
		   
case when BAB_INSTORE_NEAR = 'N' then 1 else 0 end as INSTORE_NEAR_N,
case when BAB_INSTORE_NEAR = 'E' then 1 else 0 end as INSTORE_NEAR_E,
case when BAB_INSTORE_NEAR = 'A' then 1 else 0 end as INSTORE_NEAR_A,
case when BAB_INSTORE_NEAR = 'R' then 1 else 0 end as INSTORE_NEAR_R,
		   
case when BAB_ONLINE_NEAR = 'N' then 1 else 0 end as ONLINE_NEAR_N,
case when BAB_ONLINE_NEAR = 'E' then 1 else 0 end as ONLINE_NEAR_E,
case when BAB_ONLINE_NEAR = 'A' then 1 else 0 end as ONLINE_NEAR_A,
case when BAB_ONLINE_NEAR = 'R' then 1 else 0 end as ONLINE_NEAR_R


from
(

select 
AA.*,
BB.TRANS_ID, BB.TRANS_BOOKED_DT, BB.BAB_ALL_NEAR, BB.BAB_INSTORE_NEAR, BB.BAB_ONLINE_NEAR
from
(select * from BABY_store_customer_list) AA
join (select distinct customer_id,TRANS_ID, TRANS_BOOKED_DT, BAB_ALL_NEAR, BAB_INSTORE_NEAR, BAB_ONLINE_NEAR
from analytics_vw..ca_cust_near where TRANS_BOOKED_DT between '2019-01-01 00:00:00' and '2019-12-31 00:00:00' and CONCEPT_FORMAT_ID = 3) BB
on AA.customer_id = BB.customer_id
--where STORE_NUMBER = 42
) xx
) YY
) ZZ
group by 1,2
order by 1,2
) NEAR_indicator
on STORE_INFORMATION.STORE_NUMBER = NEAR_indicator.STORE_NUMBER

join

--customer online sales

(
select 
STORE_NUMBER,
extract(week from TRANS_BOOKED_DT) weeks,
count(distinct XX.customer_id) online_customer_counts,
count(distinct order_trans_id) online_transaction_counts,
sum(gross_sales) online_gross_sales

from
(
select distinct customer_id, order_trans_id, trans_booked_dt, gross_sales
from SK_CUSTOMER_5_YEARS_2019_BABY
where INSTORE_ONLINE_IND = 'online'
) xx
join
BABY_store_customer_list YY
on XX.customer_id = YY.customer_id
group by 1,2
order by 1,2
) online_sales

on NEAR_indicator.STORE_NUMBER = online_sales.STORE_NUMBER
and NEAR_indicator.weeks = online_sales.weeks

--customer instore sales
join
(
select 
STORE_NUMBER,
extract(week from TRANS_BOOKED_DT) weeks,
count(distinct XX.customer_id) instore_customer_counts,
count(distinct order_trans_id) instore_transaction_counts,
sum(gross_sales) instore_gross_sales

from
(
select distinct customer_id, order_trans_id, trans_booked_dt, gross_sales
from SK_CUSTOMER_5_YEARS_2019_BABY
where INSTORE_ONLINE_IND = 'instore'
) xx
join
BABY_store_customer_list YY
on XX.customer_id = YY.customer_id
group by 1,2
order by 1,2
) instore_sales

on instore_sales.STORE_NUMBER = online_sales.STORE_NUMBER
and instore_sales.weeks = online_sales.weeks

join
(
select 
distinct weeks, CALENDAR_SEASON_CD
from
(
select
distinct
extract(week from CALENDAR_DT) as weeks,
CALENDAR_SEASON_CD,
row_number() over (partition by weeks order by weeks) as rn 
from EDW_MCF_VW..CALENDAR_DATE 
--where CALENDAR_DT between '2019-01-01 00:00:00' and '2019-12-31 00:00:00'
order by 1
) xx
where rn=1
order by 1
) SEASONALITY

on instore_sales.weeks = SEASONALITY.weeks
order by 1,2;
