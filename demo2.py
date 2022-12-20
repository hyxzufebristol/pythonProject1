# -*- coding: utf-8 -*- 
# @Time : 2022/12/17 12:08 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from config import *


df = pd.read_csv("./data/clean/df_clean.csv", delimiter=",")
reviews = pd.read_csv("./data/raw/reviews.csv", delimiter=",")


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
cat_df_km.to_csv("./data/clean/listing_cluster.csv")

df['host_is_superhost'] = df['host_is_superhost'].map({'t':1,'f':0})


reviews.drop(columns=['id'],inplace=True)
reviews = reviews.rename(columns={'listing_id': 'id'})
df_concat = pd.merge(df,reviews,how="inner",on='id')
df_concat.to_csv("./data/clean/df_concat.csv", index=False)



