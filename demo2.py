# -*- coding: utf-8 -*- 
# @Time : 2022/12/17 12:08 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from config import *


df = pd.read_csv(data_out + "df_clean.csv", delimiter=",")

# ----------- 地图代码 ------------
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


# ----- find the most hot one ---------


# nlp
# ['neighborhood_overview','host_about','host_verifications']



# ----- 聚类 --------
features1 = df.select_dtypes(exclude='object')
# 确定聚类数量
wcss = []

# Fit the model for a range of values for the number of clusters
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(features1)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
plt.plot(range(2, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# 依据确定值进行类别选取
kmeans=KMeans(n_clusters=6,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_pred=kmeans.fit_predict(features1)
print(y_pred)

# 聚类结果
cat_df_km=df.copy()
cat_df_km['km_result']=y_pred
