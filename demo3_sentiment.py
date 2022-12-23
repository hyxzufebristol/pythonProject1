#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/22 20:58
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
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

reviews_details = df_concat[['id', 'name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments']]

host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
    name="number_of_reviews")


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



import re


def is_english(row):
    return bool(re.match(r'^[a-zA-Z\s]+$', row))


reviews_details['is_english'] = reviews_details.apply(lambda row: is_english(row['comments']), axis=1)
reviews_details = reviews_details.loc[reviews_details['is_english'] == True]


scorer = SentimentIntensityAnalyzer()


def calculate_sentiment(comment):
    return (scorer.polarity_scores(comment)['compound'])


reviews_details.loc[:, 'sentiment'] = reviews_details['comments'].apply(calculate_sentiment)

# create positive comments
df_positive = reviews_details[reviews_details['sentiment'] > 0]
# create negative comments
df_negative = reviews_details[reviews_details['sentiment'] < 0]

# 先选取正向文本进行测试
texts = df_positive.comments.tolist()
# 文本向量化的提取方法
# 提取方法1
texts = texts[:1000]
from sklearn.feature_extraction.text import TfidfVectorizer
# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
# just send in all your docs here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)
X = tfidf_vectorizer_vectors.toarray()
Y = df_positive['sentiment'][:1000].astype("float64")


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,roc_curve,auc
from sklearn.tree import DecisionTreeClassifier

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.2)
# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)



# 序列相关性分析
import numpy as np
from scipy.stats import pearsonr, spearmanr

data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
r, p = pearsonr(data1, data2)
print('Pearson correlation coefficient:', r)
print('p-value:', p)