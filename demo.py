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


# zero step
df = pd.read_csv(data_in + "listings.csv", delimiter=",", dtype="unicode")

# ----------------- map1 -------------
# lats2018 = data['latitude'].tolist()
# lons2018 = data['longitude'].tolist()
# locations = list(zip(lats2018, lons2018))
#
# map1 = folium.Map(location=[52.3680, 4.9036], zoom_start=11.5)
# FastMarkerCluster(data=locations).add_to(map1)


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


df_listings_drop5.to_csv(data_out+"df_clean.csv", index=False)




