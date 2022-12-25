# -*- coding: utf-8 -*- 
# @Time : 2022/12/17 12:08 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from config import *

df = pd.read_csv("./data/clean/df_concat.csv", nrows=1000)
df = reduce_mem_usage(df)


def kmeans_process(df):
    """
    使用两种不同的方式来判断族的个数，并将确定的族个数用于后续的label确定
    值得注意的是：需要确定X矩阵的columns标签，有哪些变量需要被考虑在内
    :param df:
    :return:
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    # Load the data
    X = df.select_dtypes(exclude='object')

    # method 1
    # Create an empty list to store the SSE values
    sse = []
    # Fit the KMeans model to the data with a range of values for the number of clusters
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Plot the SSE values
    plt.plot(range(1, 10), sse)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

    # method 2
    # Create an empty list
    # Create an empty list to store the silhouette scores
    scores = []

    # Fit the KMeans model to the data with a range of values for the number of clusters
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.predict(X)

        # Calculate the silhouette score for the current number of clusters
        score = silhouette_score(X, labels)
        scores.append(score)

    # Plot the silhouette scores
    plt.plot(range(2, 10), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()


def kmeans_cluter(df):
    """
    使用上面的可视化函数来确定最终的类别数量
    :param df:
    :return:
    """
    # 依据确定值进行类别选取
    X = df.select_dtypes(exclude='object')
    kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, random_state=0)
    y_pred = kmeans.fit_predict(X)
    # print(y_pred)

    # 聚类结果
    cat_df_km = df.copy()
    cat_df_km['km_result'] = y_pred
    # cat_df_km.to_csv("./data/clean/listing_cluster.csv")
    return cat_df_km

def dbscan_cluter(df):
    """
    最终核对结果发现还是上面的方式比较好用
    这个确定的类别数量并不合理
    :param df:
    :return:
    """
    from sklearn.cluster import DBSCAN
    X = df.select_dtypes(exclude='object')
    # Create a DBSCAN object
    dbscan = DBSCAN()
    # Fit the DBSCAN object to the data
    dbscan.fit(X)
    # Predict the cluster labels for the data
    labels = dbscan.labels_
    # Find the unique values in the array
    unique_values, counts = np.unique(labels, return_counts=True)
    print("unique_values is {} and counts is {}".format(unique_values, counts))

    df['cluters'] = labels
    return df


kmeans_process(df)
df_cluter = kmeans_cluter(df)
# df_cluter = dbscan_cluter(df)