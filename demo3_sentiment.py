#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/22 20:58
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import *
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

"""
这部分内容对词语进行了清洗但是并没有保存？

"""
df_concat = pd.read_csv('./data/clean/df_concat.csv',nrows=10000)
df_concat = reduce_mem_usage(df_concat)
import pandas as pd


# 2. the most top 30 words in newreview data
# the reviews are mainly from the 'comment' column.

reviews_details = df_concat[['id', 'name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments', 'review_scores_value']]

host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
    name="number_of_reviews")

def comment_processing(reviews_details):
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
    return reviews_details

reviews_details = comment_processing(reviews_details)


import re


def is_english(row):
    return bool(re.match(r'^[a-zA-Z\s]+$', row))


reviews_details['is_english'] = reviews_details.apply(lambda row: is_english(row['comments']), axis=1)
reviews_details = reviews_details.loc[reviews_details['is_english'] == True]

# ---------------------------------------- 1. 针对文本情感分析分类----------------------
# --------------- 计算方法1 --------------
# 偏向于回归
scorer = SentimentIntensityAnalyzer()
def calculate_sentiment(comment):
    return (scorer.polarity_scores(comment)['compound'])
reviews_details.loc[:, 'sentiment'] = reviews_details['comments'].apply(calculate_sentiment)
# create positive comments
df_positive = reviews_details[reviews_details['sentiment'] > 0]
# create negative comments
df_negative = reviews_details[reviews_details['sentiment'] < 0]

# --------------- 计算方法2 --------------
# 使用review_scores_value 做多分类



# 先选取正向文本进行测试
texts = df_positive.comments.tolist()
# 文本向量化的提取方法
# tfidf 提取方式
texts = texts[:1000]
# TfidfVectorizer 的效果就是CountVectorizer、TfidfTransformer的结合体
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
# just send in all your docs here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)

# 对文本进行词频统计
vectorizer = CountVectorizer()
vectorizer.fit_transform(texts)
top_words = vectorizer.get_feature_names()
print("Top words: {}".format(top_words[:100]))


X = tfidf_vectorizer_vectors.toarray()
Y_reg = df_positive['sentiment'][:1000].astype("float64")
Y=Y_reg
Y_clf = df_positive.review_scores_value[:1000]
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,roc_curve,auc
from sklearn.tree import DecisionTreeClassifier

X_train,X_test,y_train,y_test=train_test_split(X, Y_clf, test_size=0.2)

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

# 可视化处理分析

fig = go.Figure([go.Bar(x=Y.value_counts().index, y=Y.value_counts().tolist())])
fig.update_layout(
    title="Values in each Sentiment",
    xaxis_title="Sentiment",
    yaxis_title="Values")
fig.show()


# 多个不同的模型进行比对测试
# --------------------- 模型2 决策树 ----------------
dt = DecisionTreeClassifier(random_state=1234)
dt.fit(X_train,y_train)
y_pred_test = dt.predict(X_test)
print("Training Accuracy score: "+str(round(accuracy_score(y_train,dt.predict(X_train)),4)))
print("Testing Accuracy score: "+str(round(accuracy_score(y_test,dt.predict(X_test)),4)))
print(classification_report(y_test, y_pred_test))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred_test)
#print('Confusion matrix\n', cm)
cm_matrix = pd.DataFrame(data=cm
                         # columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'],
                         # index=['Predict Negative', 'Predict Neutral', 'Predict Positive']
                         )
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

# 模型2
# 后面再增加几个分类模型和一个集成模型，来对比一下效果可以（不一定必要）





# -------------------------------- 2. 使用随机森林构建权重分析 -----------------------
# 这一部分内容按照所有特征列，并将review_scores_value作为最终的分类目标
