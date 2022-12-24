#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/22 20:58
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score, \
    roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import re
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

"""
这部分内容对词语进行了清洗但是并没有保存？

"""
df_concat = pd.read_csv('./data/clean/df_concat.csv', nrows=20000)
df_concat = reduce_mem_usage(df_concat)

# 2. the most top 30 words in newreview data
# the reviews are mainly from the 'comment' column.

reviews_details = df_concat[
    ['id', 'name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments', 'review_scores_value']]

#
host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
    name="number_of_reviews")


def col_comment_processing(reviews_details):
    """
    针对评论数据当中的comments 列进行处理
    要求：
    1. 实现文本评论的清洗
    2. 仅保留英文评论
    :param reviews_details:
    :return:
    """
    # and the basic pre processing code for comment. take out empty comments. 并对评论的各种格式进行修改，其中存在很多无意义的连接和我们并不希望进行统计的内容
    reviews_details = reviews_details[reviews_details['comments'].notnull()]
    # remove numbers
    reviews_details['comments'] = reviews_details['comments'].str.replace('\d+', '')
    # all to lowercase
    reviews_details['comments'] = reviews_details['comments'].str.lower()
    # remove windows new line
    reviews_details['comments'] = reviews_details['comments'].str.replace('\r\n', "")
    # remove stopwords (from nltk library)
    stop_english = stopwords.words("english")
    reviews_details['comments'] = reviews_details['comments'].apply(lambda x: " ".join([i for i in x.split()
                                                                                        if i not in (stop_english)]))
    # remove punctuation
    reviews_details['comments'] = reviews_details['comments'].str.replace('[^\w\s]', " ")
    # replace x spaces by one space
    reviews_details['comments'] = reviews_details['comments'].str.replace('\s+', ' ')

    def is_english(row):
        return bool(re.match(r'^[a-zA-Z\s]+$', row))

    reviews_details['is_english'] = reviews_details.apply(lambda row: is_english(row['comments']), axis=1)
    reviews_details = reviews_details.loc[reviews_details['is_english'] == True]

    return reviews_details

# 对comment列进行完整处理
reviews_details = col_comment_processing(reviews_details)


def comment_to_sentimentStrength(reviews_details):
    """
    通过对文本情感强度进行分析，获得对sentiment进行一个计算，增加了一个新的列，能够用于回归任务
    :param reviews_details:
    :return:
    """
    scorer = SentimentIntensityAnalyzer()  # 将comment

    def calculate_sentiment(comment):
        return (scorer.polarity_scores(comment)['compound'])

    # TODO： 这部分是构建出来的完整数据集
    reviews_details.loc[:, 'sentiment'] = reviews_details['comments'].apply(calculate_sentiment)
    return reviews_details

# 获得sentiment列，也就是文本情感强度
reviews_details = comment_to_sentimentStrength(reviews_details)

# create positive comments
df_positive = reviews_details[reviews_details['sentiment'] > 0]
# create negative comments
df_negative = reviews_details[reviews_details['sentiment'] < 0]

# ---------------------------------------- 1. 针对文本情感分析分类----------------------
#

# 先选取正向文本进行测试
texts = df_positive.comments.tolist()
# 文本向量化的提取方法
# tfidf 提取方式
texts = texts[:1000]
# TfidfVectorizer 的效果就是CountVectorizer、TfidfTransformer的结合体
# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
# just send in all your docs here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)
# 构建用于文本情感分类的xy数据
X = tfidf_vectorizer_vectors.toarray()
Y_clf = df_positive.review_scores_value[:1000]


# 输出top100 关键词
def top_words_importance_plt(texts):
    """
    使用tf-idf对输入文本内容进行处理并输出top 100 的关键词
    :param texts:
    :return:
    """
    # # 对文本进行词频统计
    # vectorizer = CountVectorizer()
    # vectorizer.fit_transform(texts)
    # top_words = vectorizer.get_feature_names()
    # print("Top words: {}".format(top_words[:100]))

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(max_features=100)
    # Fit the vectorizer to the texts
    vectorizer.fit(texts)
    # Extract the top 100 features and their importance weights
    features = vectorizer.get_feature_names()
    importances = vectorizer.idf_
    # Zip the features and importance weights together
    keywords = zip(features, importances)
    # Sort the keywords by importance weight in descending order
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    # Print the top 100 keywords and their importance weights
    for keyword, importance in keywords[:100]:
        print(f'{keyword}: {importance:.2f}')

# 输出top100 关键词
top_words_importance_plt(texts)




def sentiment_clf_rfc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ----------------- 模型1 随机森林 -----------------
    # Create a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    # Evaluate the performance of the classifier
    accuracy = clf.score(X_test, y_test)
    print('Accuracy of classification:', accuracy)

    # 文本情感分析做分类，如果需要做回归的话用下面的代码，但是基本上都是做分类的
    # Y_clf = df_positive.review_scores_value[:1000]
    # X_train,X_test,y_train,y_test=train_test_split(X, Y_reg, test_size=0.2)
    # reg = RandomForestRegressor()
    # reg.fit(X_train, y_train)
    # accuracy = reg.score(X_test, y_test)
    # print('R2 of regression:', accuracy)


# sentiment_clf_rfc(X, Y_clf)


def sentiment_clf_dt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # --------------------- 模型2 决策树 ----------------
    dt = DecisionTreeClassifier(random_state=1234)
    dt.fit(X_train, y_train)
    y_pred_test = dt.predict(X_test)
    print("Training Accuracy score: " + str(round(accuracy_score(y_train, dt.predict(X_train)), 4)))
    print("Testing Accuracy score: " + str(round(accuracy_score(y_test, dt.predict(X_test)), 4)))
    print(classification_report(y_test, y_pred_test))

    cm = confusion_matrix(y_test, y_pred_test)
    # print('Confusion matrix\n', cm)
    cm_matrix = pd.DataFrame(data=cm
                             # columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'],
                             # index=['Predict Negative', 'Predict Neutral', 'Predict Positive']
                             )
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()


# sentiment_clf_dt(X, Y_clf)
# 模型2
# 后面再增加几个分类模型和一个集成模型，来对比一下效果可以（不一定必要）


# -------------------------------- 2. 使用随机森林构建权重分析 -----------------------
# 这一部分内容按照所有特征列,全部需要进行转化，并且不会用到有关NLP部分的col，并将review_scores_value作为最终的分类目标
# 最终处理完成的数据集是reviews_details
# 数据集需要重新更换成，并且构建两个随机森林：
# 1.找到 average monthly sentiment最大的100家home做随机森林 name
# 2.找到 average profit per month 最大的100家host做随机森林 host_name
# 注意：
# 1. 为了能够完成处理上述使用的reviews_details是缩小版本的，因为做了文本情感分类。
#    所以这个地方的需要重新从df_concat 当中构建
# 2. 由于前面代码的混乱，导致数据问题，所以我在这里只实现必要的特征，需要添加之前处理的
#    特征请替换最开始读取的特征即可，然后在下面选择特征列的位置进行确认
# Convert the date column to a datetime data type
df_concat['date'] = pd.to_datetime(df_concat['date'])

# Extract the year, month, and day from the date column
df_concat['year'] = df_concat['date'].dt.year
df_concat['month'] = df_concat['date'].dt.month
df_concat['day'] = df_concat['date'].dt.day
df_concat.loc[:, 'sentiment'] = df_concat['comments'].apply(calculate_sentiment)

# 构建特征
numerical_fea = list(df_concat.select_dtypes(exclude=['category']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(df_concat.columns)))
selected_cols = numerical_fea + [""]
# ---------------------------- 任务数据集构建 --------------------------
# 针对任务一
# Group the data by home and month
grouped_data = df_concat.groupby(['name', 'month'])
# Calculate the average monthly sentiment for each group
average_sentiment = grouped_data['sentiment'].mean()
# Select the top 100 rows based on the average monthly sentiment
top_homes = average_sentiment.nlargest(100)
# Print the top 100 homes with the largest average monthly sentiment
print(top_homes)
# 构建list
tuple_list = top_homes.index.to_list()
# Extract the first element of each tuple and construct a new list
new_list = [t[0] for t in tuple_list]
# Filter the dataframe based on the list of values
filtered_df1 = df_concat[df_concat['name'].isin(new_list)]

# 针对任务二
# Group the data by host and month
grouped_data = df_concat.groupby(['host_name', 'month'])
# Calculate the average profit per month for each group
average_profit = grouped_data['profit_per_month'].mean()
# Select the top 100 rows based on the average profit per month
top_hosts = average_profit.nlargest(100)
# Print the top 100 hosts with the largest average profit per month
print(top_hosts)
# 构建list
tuple_list = top_hosts.index.to_list()
# Extract the first element of each tuple and construct a new list
new_list = [t[0] for t in tuple_list]
# Filter the dataframe based on the list of values
filtered_df2 = df_concat[df_concat['host_name'].isin(new_list)]

# 构建Xy
# TODO：将filtered_df1和filtered_df2 对下面的df进行替换就可以进行计算了
df = df_concat.select_dtypes(exclude='category')
X = df.drop('review_scores_value', axis=1)
y = df.review_scores_value[:20000]  # 20000是最开始选择到的，28行


def rfc_process(X, y):
    """
    随机森铃实现分类，同时输出最终的权重比例
    :param X:
    :param y:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 模型1
    # Create a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the performance of the classifier
    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')

    # Get the feature importances
    importances = clf.feature_importances_
    feature_names = X.columns

    # Sort the feature importance ranks in descending order
    sorted_importances = sorted(zip(importances, feature_names), reverse=True)

    # Print the sorted feature importance ranks with feature names
    print('Feature importance ranks:')
    for i, (importance, feature_name) in enumerate(sorted_importances):
        print(f'{i + 1}: {feature_name} ({importance:.2f})')
