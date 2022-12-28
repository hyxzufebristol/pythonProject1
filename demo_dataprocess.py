# -*- coding: utf-8 -*- 
# @Time : 2022/11/18 16:35 
# @Author : YeMeng 
# @File : demo_dataprocess.py
# @contact: 876720687@qq.com


import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
import missingno as msno
from matplotlib import pyplot as plt
from config import *
from math import asin, sin, cos, sqrt
import missingno as msno
import numpy as np
import pandas as pd
from config import *
import warnings

warnings.filterwarnings('ignore')


def df_listings_processing(df_listings):
    """
    对数据进行清洗过程，包含特征工程构建新特征的内容的部分
    :param df_listings:
    :return:
    """
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
    df_listings_drop0 = df_listings.drop(columns=drops)
    df_listings_drop1 = df_listings_drop0.dropna(
        subset=['reviews_per_month', 'review_scores_rating', 'review_scores_accuracy'])
    # delete the rows that contain missing values of amenities
    df_listings_drop2 = df_listings_drop1.dropna(subset=['bathrooms_text', 'bedrooms', 'beds'])
    df_listings_drop3 = df_listings_drop2.dropna(axis=1, how='all')
    df_listings_drop4 = df_listings_drop3.dropna(subset=['description', 'neighborhood_overview', 'host_about'])
    df_listings_drop5 = df_listings_drop4.dropna(subset=['host_response_time', 'host_response_rate'])

    def format_price(price):
        return (float(price.replace('$', '').replace(',', '')))

    df_listings_drop5['price_$'] = df_listings_drop5['price'].apply(format_price).astype("float32")
    df_listings_drop5['profit_per_month'] = df_listings_drop5['price_$'] * df_listings_drop5['reviews_per_month'] / \
                                            df_listings_drop5['review_scores_rating']

    # 处理host相关变量
    # replace 't' to '1', and 'f' to '0' in the columns of 'host_is_superhost','host_has_profile_pic','host_identity_verified','instant_bookable'
    # Convert 't', 'f' to 1, 0
    tf_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for tf_col in tf_cols:
        df_listings_drop5[tf_col] = df_listings_drop5[tf_col].map({'t': 1, 'f': 0})

    # as for the column of 'host_response_rate', Converts 'string' to 'int' format
    df_listings_drop5['host_response_rate'] = df_listings_drop5['host_response_rate'].str.strip('%').astype(float) / 100

    # Converts 'string' to 'int' format, classify 'host_response_time' into 4 levels
    df_listings_drop5['host_response_time'] = df_listings_drop5['host_response_time'].map({'within an hour': 100,
                                                                                           'within a few hours': 75,
                                                                                           'within a day': 50,
                                                                                           'a few days or more': 25})
    df_listings = df_listings_drop5
    ##处理 reviews相关变量
    # calculate average reviews_score
    df_listings['review_scores_avg6'] = (df_listings['review_scores_accuracy'] +
                                         df_listings['review_scores_cleanliness'] +
                                         df_listings['review_scores_checkin'] +
                                         df_listings['review_scores_communication'] +
                                         df_listings['review_scores_location'] +
                                         df_listings['review_scores_value']) / 6

    # 增加筛选方法当中需要要求distance在一定的距离限制范围值之内
    # 错误写法
    # df['distance']=2*6371*asin(
    #     sqrt(
    #         sin((df['latitude'] + 37.82) / 2) ** 2 +
    #         cos(df['latitude']) * cos(df['latitude']) * (sin((df['longitude']-144.96)/2)**2)
    #          )
    # )
    df_listings['distance'] = 2 * 6371 * (
        (
                ((df_listings['latitude'] + 37.82) / 2).apply(sin) ** 2 +
                (df_listings['latitude']).apply(cos) * (df_listings['latitude']).apply(cos) * (
                            ((df_listings['longitude'] - 144.96) / 2).apply(sin) ** 2)
        ).apply(sqrt)
    ).apply(asin)

    return df_listings


## ------------------------------- 处理reviews -------------------------
def df_review_processing(df_reviews):
    # delete comments missing value
    # 由于我们的问题focus on消费者的评论信息，因此值保留评论信息完整的所有数据
    df_reviews = df_reviews[~df_reviews['comments'].isna()]

    # ---------------- 时间类数据处理 -------
    # Convert the date column to a datetime data type
    df_reviews['date'] = pd.to_datetime(df_reviews['date'])
    # Extract the year, month, and day from the date column
    df_reviews['year'] = df_reviews['date'].dt.year
    df_reviews['month'] = df_reviews['date'].dt.month
    df_reviews['day'] = df_reviews['date'].dt.day

    # drop rows that 'listing_id' doesn't exist in the 'id' of 'listings' and create a new dataset'concat'
    # merge dataset 'listings' and 'reviews' with the same 'id'
    df_reviews.drop(columns=['id'], inplace=True)
    df_reviews = df_reviews.rename(columns={'listing_id': 'id'})
    return df_reviews


if __name__ == '__main__':
    # zero step
    df_listings = pd.read_csv('./data/raw/listings.csv')
    # df_listings = pd.read_csv('./data/raw/listings.csv', nrows=10000)
    df_listings = reduce_mem_usage(df_listings)
    df_reviews = pd.read_csv('./data/raw/reviews.csv')
    # df_reviews = pd.read_csv('./data/raw/reviews.csv', nrows=10000)
    df_reviews = reduce_mem_usage(df_reviews)

    # # 缺失值可视化
    # msno.matrix(df_listings, labels=True)
    # # 热力图可视化
    # msno.heatmap(df_listings)
    # # 树状图可视化
    # msno.dendrogram(df_listings)

    # -------------------- 对处理完毕的数据进行合并 -------------
    # ----------------------- 合并dataframe -----------------
    # 这部分内容是完成了listing 和reviews两个数据集的合并，并将其保存到了df_concat.csv 文件当中
    df_listings = df_listings_processing(df_listings)
    df_reviews = df_review_processing(df_reviews)
    df_concat = pd.merge(df_listings, df_reviews, how="inner", on='id')

    df_concat.to_csv("./data/clean/df_concat.csv", index=False)
    # df_listings.to_csv("./data/clean/df_listings.csv", index=False)
    print("Processing success!")
