# -*- coding: utf-8 -*- 
# @Time : 2022/12/19 10:17 
# @Author : YeMeng 
# @File : vis2.py 
# @contact: 876720687@qq.com
# 新增六个项目内容
# 合并完成内容在concat数据集当中
import re

import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from config import *

# import nltk
# nltk.download("stopwords")
# stop_english = stopwords.words("english")

df = pd.read_csv("./data/clean/df_concat.csv")
df = reduce_mem_usage(df)

# 2. the most top 30 words in newreview data
# the reviews are mainly from the 'comment' column.
reviews_details = df[['name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments']]

host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
    name="number_of_reviews")
# you can show something by the code below.
# host_reviews.head()
# reviews_details.comments.head()
# reviews_details.comments.values[1]

# and the basic pre processing code for comment. take out empty comments. 并对评论的各种格式进行修改，其中存在很多无意义的连接和我们并不希望进行统计的内容
reviews_details = reviews_details[reviews_details['comments'].notnull()]
# remove numbers
reviews_details['comments'] = reviews_details['comments'].str.replace('\d+', '')
# all to lowercase
reviews_details['comments'] = reviews_details['comments'].str.lower()
# remove windows new line
reviews_details['comments'] = reviews_details['comments'].str.replace('\r\n', "")
# remove stopwords (from nltk library)
# # 加载停用词
# reviews_details['comments'] = reviews_details['comments'].apply(lambda x: " ".join([i for i in x.split()
#                                                                                     if i not in (stop_english)]))
# remove punctuation
reviews_details['comments'] = reviews_details['comments'].str.replace('[^\w\s]', " ")
# replace x spaces by one space
reviews_details['comments'] = reviews_details['comments'].str.replace('\s+', ' ')


def is_english(row):
    return bool(re.match(r'^[a-zA-Z\s]+$', row))


reviews_details['is_english'] = reviews_details.apply(lambda row: is_english(row['comments']), axis=1)
reviews_details = reviews_details.loc[reviews_details['is_english'] == True]

print(reviews_details.comments.values[2])

# find the most used words using the CountVectorizer() function of sklearn.
texts = reviews_details.comments.tolist()

# 文本向量化的提取方法
# 提取方法1
texts = texts[:1000]
from sklearn.feature_extraction.text import TfidfVectorizer
# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
# just send in all your docs here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)
X = tfidf_vectorizer_vectors.toarray()
Y = df['Sentiment'][:7000]



# 方法2
vec = CountVectorizer().fit(texts)
bag_of_words = vec.transform(texts)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

cvec_df = pd.DataFrame.from_records(words_freq, columns=['words', 'counts']).sort_values(by="counts", ascending=False)

# # 柱状图展示
# # cvec_df.head(10).plot.barh(figsize=(15, 3), width=0.4)
# plt.barh(cvec_df.head(10).words.to_list(),
#          cvec_df.head(10).counts.to_list()
#          )
# plt.show()
#
# # 词云展示
# cvec_dict = dict(zip(cvec_df.words, cvec_df.counts))
#
# wordcloud = WordCloud(width=800, height=400)
# wordcloud.generate_from_frequencies(frequencies=cvec_dict)
# plt.figure(figsize=(20, 10))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()





# 3. Perimeter of number of listings per host.
# 我们需要找到最终入围的host有哪几个人？这个可以说是爱彼迎的优质会员了
df = pd.read_csv("./data/clean/df_concat.csv")
df = reduce_mem_usage(df)

# host_names = df[['host_name']].drop_duplicates()
# host_names = host_names.host_name.str.lower().tolist()
# print(len(host_names))

# 你需要什么内容这里选择一下就能显示了
# col_describe = ["host_is_superhost","host_listings_count","accommodates","bedrooms","beds"]# 太多了，需要自己选
# df.groupby(by="host_name")[col_describe].sum().describe()
df.groupby(by="host_name").sum().describe()










# 4. 计算从1-12⽉每个⽉的不同roomtype的数量（按照⽉份来计算）
# 从new review的⽂档⾥⾯，列出 id， room type，date 先将data⾥⾯拆分为year month day计算/拿出month另起⼀列计算
df4 = df[["id", "room_type", "", "date"]]
df4['date'] = pd.to_datetime(df4['date'], format='%Y-%m-%d')
df4["data_year"] = df4.date.dt.year
df4["data_month"] = df4.date.dt.month
df4["data_day"] = df4.date.dt.day
df_grouped = df4.groupby(by="data_month")["room_type"].value_counts()
# Unstack the data to get it into a format that can be plotted
df_unstacked = df_grouped.unstack()

## Create a figure and a 3D axis
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for line in df_unstacked.columns:
#     # Plot the data set as a 2D line chart
#     ax.plot(df_unstacked.index, df_unstacked[line])
# # ax.plot(df_unstacked.index, df_unstacked["Entire home/apt"])
# # ax.plot(df_unstacked.index, df_unstacked["Private room"])
# # ax.plot(df_unstacked.index, df_unstacked["Shared room"])
# # ax.plot(df_unstacked.index, df_unstacked["Hotel room"])
# # Show the plot
# plt.show()


# Create a figure and a 2D axis
fig = plt.figure()
ax = plt.axes()
# ax.plot(df_unstacked.index, df_unstacked["Entire home/apt"])
# ax.plot(df_unstacked.index, df_unstacked["Private room"])
ax.plot(df_unstacked.index, df_unstacked["Shared room"])
ax.plot(df_unstacked.index, df_unstacked["Hotel room"])
# Show the plot
plt.show()












# 5.Show计算不同⽉份的price情况
# 从new review的⽂档⾥⾯，列出 id，price，date\
df5 = df[["id", "price_$", "date"]]
df5['date'] = pd.to_datetime(df5['date'], format='%Y-%m-%d')
df5["data_year"] = df5.date.dt.year
df5["data_month"] = df5.date.dt.month
df5["data_day"] = df5.date.dt.day

df5['price_$'] = df5['price_$'].astype("float64")
df_grouped2 = df5.groupby(by="data_month")["price_$"].sum()
df_grouped2.plot.barh()
plt.show()












# 6.room type和price 直接的关系，横坐标是month
df6 = df[["id", "room_type", "price_$", "date"]]
df6['date'] = pd.to_datetime(df6['date'], format='%Y-%m-%d')
df6["data_year"] = df6.date.dt.year
df6["data_month"] = df6.date.dt.month
df6["data_day"] = df6.date.dt.day
df6['price_$'] = df6['price_$'].astype("float64")
df_grouped3 = df6.groupby(["data_month", "room_type"])['price_$'].mean()
# for analysis
# df_grouped3 = df6.groupby(["data_month", "room_type"])
# aggregated = df_grouped3.agg({'price_$': ['mean', 'max']})

# this could be the output.
df_unstacked = df_grouped3.unstack()
