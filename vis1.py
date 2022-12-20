# -*- coding: utf-8 -*- 
# @Time : 2022/12/18 22:10 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from config import *


df = pd.read_csv("./data/clean/df_clean.csv", delimiter=",")

# TODO:visulization
# ----------- 地图代码 ------------
# 需要走墙，国内不出图
# # zero step
# data = pd.read_csv(data_out + "df_clean.csv", delimiter=",", dtype="unicode")
#
#
# lats2018 = data['latitude'].tolist()
# lons2018 = data['longitude'].tolist()
# locations = list(zip(lats2018, lons2018))
#
# map1 = folium.Map(location=[52.3680, 4.9036], zoom_start=11.5)
# FastMarkerCluster(data=locations).add_to(map1)

import pandas as pd
from folium.plugins import FastMarkerCluster
from haversine import haversine

from config import reduce_mem_usage

listings = pd.read_csv("./data/clean/df_concat.csv")
listings = reduce_mem_usage(listings)
# loc1 = s.loc[:, ["latitude", "longitude"]].values[0].tolist()

# the place that we want to go.
destination = [-37.82, 144.96]


def cal_distance(la, lon):
    loc2 = [la, lon]
    return haversine(destination, loc2)


listings['distance'] = listings[['latitude', "longitude"]].apply(
    lambda x: cal_distance(x['latitude'], x['longitude'])
    , axis=1)

# draw maps
lats = listings['latitude'].tolist()
lons = listings['longitude'].tolist()
locations = list(zip(lats, lons))
import folium
map1 = folium.Map(location=destination, zoom_start=11.5)
folium.Circle(radius=5000,location=destination, color="blue", fill=False).add_to(map1)
# FastMarkerCluster(data=locations).add_to(map1)
df_loc = listings.loc[:["latitude","longitude"]].values.tolist()
marker_cluster = FastMarkerCluster().add_to(map1)
for loc in df_loc:
    folium.Marker(location=loc,icon=None).add_to(marker_cluster)


# ------------ 分析 ------------
#
freq = df['room_type']. value_counts().sort_values(ascending=True)
freq.plot.barh(figsize=(15, 3), width=0.7, color = ["g","b","r","y"])
plt.show()

# top host
rank_data = df.groupby(['host_name'])['profit_per_month','reviews_per_month'].apply(sum)
rank_data = rank_data.sort_values('profit_per_month', ascending=False)
rank_data.iloc[:10,:1].plot.barh(figsize=(15, 3), width=0.4)
plt.show()



# review 一系列都是得分能够直接获得消费者对这家店的评价
review_score = df[df['number_of_reviews']>=10]

fig = plt.figure(figsize=(20,15))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

ax1 = fig.add_subplot(321)
feq=review_score['review_scores_location'].value_counts().sort_index()
ax1=feq.plot.bar(color='b', width=0.7, rot=0)
#ax1.tick_params(axis = 'both', labelsize = 16)
plt.title("Location", fontsize=24)
plt.ylabel('Number of listings', fontsize=14)
plt.xlabel('Average review score', fontsize=14)

ax2 = fig.add_subplot(322)
feq=review_score['review_scores_cleanliness'].value_counts().sort_index()
ax2=feq.plot.bar(color='b', width=0.7, rot=0)
plt.title("Cleanliness", fontsize=24)
plt.ylabel('Number of listings', fontsize=14)
plt.xlabel('Average review score', fontsize=14)

ax3 = fig.add_subplot(323)
feq=review_score['review_scores_value'].value_counts().sort_index()
ax3=feq.plot.bar(color='b', width=0.7, rot=0)
plt.title("Value", fontsize=24)
plt.ylabel('Number of listings', fontsize=14)
plt.xlabel('Average review score', fontsize=14)

ax4 = fig.add_subplot(324)
feq=review_score['review_scores_communication'].value_counts().sort_index()
ax4=feq.plot.bar(color='b', width=0.7, rot=0)
plt.title("Communication", fontsize=24)
plt.ylabel('Number of listings', fontsize=14)
plt.xlabel('Average review score', fontsize=14)

ax5 = fig.add_subplot(325)
feq=review_score['review_scores_checkin'].value_counts().sort_index()
ax5=feq.plot.bar(color='b', width=0.7, rot=0)
plt.title("Arrival", fontsize=24)
plt.ylabel('Number of listings', fontsize=14)
plt.xlabel('Average review score', fontsize=14)

ax6 = fig.add_subplot(326)
feq=review_score['review_scores_accuracy'].value_counts().sort_index()
ax6=feq.plot.bar(color='b', width=0.7, rot=0)
plt.title("Accuracy", fontsize=24)
plt.ylabel('Number of listings', fontsize=14)
plt.xlabel('Average review score', fontsize=14)

plt.tight_layout()
plt.show()



