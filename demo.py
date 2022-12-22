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

# zero step
df = pd.read_csv("./data/raw/listings.csv", delimiter=",", dtype="unicode")


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
]
df_listings_drop0 = df.drop(columns=drops)
msno.matrix(df_listings_drop0, labels=True)

df_listings_drop1 = df_listings_drop0.dropna(subset=['reviews_per_month','review_scores_rating','review_scores_accuracy'])
#delete the rows that contain missing values of amenities
df_listings_drop2 = df_listings_drop1.dropna(subset=['bathrooms_text','bedrooms','beds'])
#delete the column containing only missing values
df_listings_drop3=df_listings_drop2.dropna(axis=1,how='all')

df_listings_drop4 = df_listings_drop3.dropna(subset=['description','neighborhood_overview','host_about'])

df_listings_drop5 = df_listings_drop4.dropna(subset=['host_response_time','host_response_rate'])

# df_listings_drop5['price']
# remove '$' of 'price'
def format_price(price):
    return(float(price.replace('$','').replace(',','')))

df_listings_drop5['price_$'] = df_listings_drop5['price'].apply(format_price)


# (homes' names of 100 highest profit_per_month)
df_listings_drop5['profit_per_month'] = df_listings_drop5['price_$'] * df_listings_drop5['reviews_per_month'].astype("float")
df_listings_drop5 = df_listings_drop5.sort_values(by=['profit_per_month'],ascending=False)


# df_listings_drop5.to_csv("./data/clean/df_clean.csv", index=False)



# 增加筛选方法当中需要要求distance在一定的距离限制范围值之内
df['distance']=2*6371*(
    (
        ((df['latitude'] + 37.82) / 2).apply(sin)**2 +
        (df['latitude']).apply(cos) * (df['latitude']).apply(cos) * (((df['longitude']-144.96)/2).apply(sin)**2)
     ).apply(sqrt)
).apply(asin)



# 一些额外的处理方法
# df['host_is_superhost'] = df['host_is_superhost'].map({'t':1,'f':0})

# df['review_scores_avg6']=(df['review_scores_accuracy']+df['review_scores_cleanliness']+df['review_scores_checkin']+df['review_scores_communication']+df['review_scores_location']+df['review_scores_value'])/6
