# -*- coding: utf-8 -*- 
# @Time : 2022/11/18 16:35 
# @Author : YeMeng 
# @File : demo.py 
# @contact: 876720687@qq.com


# 处理缺失值
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
import missingno as msno
from matplotlib import pyplot as plt
from config import *
from math import *
import missingno as msno
import numpy as np
import pandas as pd

# zero step
df_listings = pd.read_csv('listings.csv')
df_reviews = pd.read_csv('reviews.csv')

# Missing value visualization
# msno.matrix(df_listings, labels=True)

drops = [
    "last_scraped",
    "calendar_last_scraped",
    "host_acceptance_rate",
    "host_total_listings_count",
    "host_neighbourhood",
    "neighbourhood",
    "picture_url",
    "host_url",
    "host_thumbnail_url",
    "host_picture_url",
    "has_availability"
]
df_listings = df_listings.drop(columns=drops)
# 热力图可视化
msno.heatmap(df_listings)
df_listings = df_listings.dropna(subset=['reviews_per_month',
                                         'review_scores_rating',
                                         'review_scores_accuracy',
                                         'bathrooms_text',
                                         'bedrooms',
                                         'beds',
                                         'description',
                                         'neighborhood_overview',
                                         'host_about',
                                         'host_response_time',
                                         'host_response_rate',
                                         'host_location'
                                         ])
# delete the column containing only missing values
df_listings = df_listings.dropna(axis=1, how='all')


# msno.matrix(df_listings, labels=True)

## 处理具体indicators of price, profit
# remove '$' of 'price'
def format_price(price):
    return (float(price.replace('$', '').replace(',', '')))


df_listings['price_$'] = df_listings['price'].apply(format_price)
df_listings[['price', 'price_$']].head()

# (homes' names of 100 highest profit_per_month), profit_per_month = price_$ * reviews_per_month/review_scores_rating
df_listings['profit_per_month'] = df_listings['price_$'] * df_listings['reviews_per_month'] / df_listings[
    'review_scores_rating']

df_listings = df_listings.sort_values(by=['profit_per_month'], ascending=False)

##处理 reviews相关变量
# calculate average reviews_score
df_listings['review_scores_avg6'] = (df_listings['review_scores_accuracy'] + df_listings['review_scores_cleanliness'] +
                                     df_listings['review_scores_checkin'] + df_listings['review_scores_communication'] +
                                     df_listings['review_scores_location'] + df_listings['review_scores_value']) / 6

##处理host相关变量
# replace 't' to '1', and 'f' to '0' in the columns of 'host_is_superhost','host_has_profile_pic','host_identity_verified','instant_bookable'

# Convert 't', 'f' to 1, 0
tf_cols = ['host_is_superhost',
           'host_has_profile_pic',
           'host_identity_verified',
           'instant_bookable']
for tf_col in tf_cols:
    df_listings[tf_col] = df_listings[tf_col].map({'t': 1, 'f': 0})

# as for the column of 'host_response_rate', Converts 'string' to 'int' format
df_listings['host_response_rate'] = df_listings['host_response_rate'].str.strip('%').astype(float) / 100

# Converts 'string' to 'int' format, classify 'host_response_time' into 4 levels
df_listings['host_response_time'] = df_listings['host_response_time'].map({
    'within an hour': 100,
    'within a few hours': 75,
    'within a day': 50,
    'a few days or more': 25})

## ------------------------------- 处理reviews -------------------------
# delete comments missing value
df_reviews = df_reviews[~df_reviews['comments'].isna()]

# drop rows that 'listing_id' doesn't exist in the 'id' of 'listings' and create a new dataset'concat'
# merge dataset 'listings' and 'reviews' with the same 'id'
df_reviews.drop(columns=['id'], inplace=True)
df_reviews = df_reviews.rename(columns={'listing_id': 'id'})
df_concat = pd.merge(df_listings, df_reviews, how="inner", on='id')

df_listings.to_csv("df_listings.csv")

# 增加筛选方法当中需要要求distance在一定的距离限制范围值之内
# df['distance']=2*6371*(
#     (
#         ((df['latitude'] + 37.82) / 2).apply(sin)**2 +
#         (df['latitude']).apply(cos) * (df['latitude']).apply(cos) * (((df['longitude']-144.96)/2).apply(sin)**2)
#      ).apply(sqrt)
# ).apply(asin)
