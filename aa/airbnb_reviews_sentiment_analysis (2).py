#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_concat=pd.read_csv('df_concat.csv')


# In[2]:


# reduce volumn of dataset stored in RAM/memory
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


# reduce 
import numpy as np
df_concat = reduce_mem_usage(df_concat)


# In[4]:


import pandas as pd
df_listings= pd.read_csv("df_listings.csv")
df_listings.shape


# In[5]:


df_concat.columns


# ## Figure 3.1 Room-type visualization of all hosts

# In[6]:


from matplotlib import pyplot as plt
freq = df_listings['room_type']. value_counts().sort_values(ascending=True)
freq.plot.barh(figsize=(15, 3), width=0.7, color = ["g","b","r","y"])
plt.show()


# ## Figure 3.2 Room-type visualization of high-profit hosts

# In[7]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from config import *

freq = df_concat['room_type']. value_counts().sort_values(ascending=True)
freq.plot.barh(figsize=(15, 3), width=0.7, color = ["g","b","r","y"])
plt.show()


# ## Figure 3.3 reviews scores visualization

# In[8]:


review_score = df_concat[df_concat['number_of_reviews']>=10]

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
plt.title("Checkin", fontsize=24)
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


# ## Figure 3.4 max,mean, min, Upper and lower quarters，standard variance of number of hosts' listings 

# In[9]:


df_concat['host_name'].value_counts().describe()


# ## Figure 3.4 Max, Mean, Min, Upper and Lower Quarters, Standard Variance of number of hosts' listings

# In[10]:


df_concat['host_name'].value_counts().describe().plot.barh()


# In[11]:


# Import libs
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Get the data for Iron Man
labels=np.array(["price_$","review_scores_avg6","availability_60","bedrooms","beds"])
stats=df_concat.loc[0,labels].values

# Make some calculations for the plot
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))

# Plot stuff
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'o-', linewidth=2)
ax.fill(angles, stats, alpha=0.25)
# ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title([df_concat.loc[0,"name"]])
ax.grid(True)

plt.show()


# In[12]:


# 新增六个项目内容
# 合并完成内容在concat数据集当中
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from config import *

# 2. the most top 30 words in newreview data
# the reviews are mainly from the 'comment' column.

# reviews_details = df_concat[['id','name', 
#                              'host_id', 
#                              'host_name', 
#                              'date', 
#                              'reviewer_id', 
#                              'reviewer_name', 
#                              'comments',
#                              'profit_per_month_x',
#                              'host_response_time',
#                              'host_response_rate',
#                              'host_is_superhost',
#                              'host_has_profile_pic',
#                              'host_identity_verified']]

# host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(name = "number_of_reviews")

# you can show something by the code below.
# host_reviews.head()
# reviews_details.comments.head()
# reviews_details.comments.values[1]


# In[13]:


reviews_details=df_concat


# ## Figure 3.6 Calculation of positive and negative sentiment

# In[14]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from config import *


# In[15]:


import nltk
nltk.download('stopwords')


# In[16]:


# and the basic pre processing code for comment. take out empty comments. 并对评论的各种格式进行修改，其中存在很多无意义的连接和我们并不希望进行统计的内容
reviews_details = reviews_details[reviews_details['comments'].notnull()]
#remove numbers
reviews_details['comments'] = reviews_details['comments'].str.replace('\d+', '')
#all to lowercase
reviews_details['comments'] = reviews_details['comments'].str.lower()
#remove windows new line
reviews_details['comments'] = reviews_details['comments'].str.replace('\r\n', "")
#remove stopwords (from nltk library)
stop_english = stopwords.words("english")
reviews_details['comments'] = reviews_details['comments'].apply(lambda x: " ".join([i for i in x.split()
                                                      if i not in (stop_english)]))
# remove punctuation
reviews_details['comments'] = reviews_details['comments'].str.replace('[^\w\s]'," ")
# replace x spaces by one space
reviews_details['comments'] = reviews_details['comments'].str.replace('\s+', ' ')

print(reviews_details.comments.values[2])


# In[17]:


import re


# In[18]:


def is_english(row):
    return bool(re.match(r'^[a-zA-Z\s]+$', row))


reviews_details['is_english'] = reviews_details.apply(lambda row: is_english(row['comments']), axis=1)
reviews_details = reviews_details.loc[reviews_details['is_english'] == True]


# In[19]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
scorer = SentimentIntensityAnalyzer()


# In[20]:


#calculate sentiments of reviews
# Delete the columns containing non-English words
def calculate_sentiment(comment):
    return(scorer.polarity_scores(comment)['compound'])

reviews_details.loc[:,'sentiment'] = reviews_details['comments'].apply(calculate_sentiment)
reviews_details[['comments','sentiment']].head()


# In[21]:


reviews_details.shape


# In[22]:


reviews_details.head()


# ## Figure 3.5 2D Density Plot of sentiment and price

# In[23]:


# Importing libs
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Create the data
price=reviews_details["price_$"]
sentiment=reviews_details['sentiment']

# Create and shor the 2D Density plot
# ax = sns.kdeplot(price, sentiment, cmap="Blues", shade=False, bw=.15, cbar=True)
# ax.set(xlabel='price', ylabel='sentiment')
# plt.show()


# In[24]:


import numpy as np
from math import *
rank_data = reviews_details.groupby(['name'])['sentiment'].mean()
from sklearn import model_selection # 提供划分训练集预测集的工具
import seaborn as sns # 提供绘制回归图象的工具	
import statsmodels.api as sm  #提供拟合回归系数的工具


# In[25]:


rank_sen=pd.read_csv('listings_name')


# In[26]:


sentiment_profit = pd.merge(rank_sen,rank_data,how="inner",on='name')


# In[27]:


import pandas as pd 
import seaborn as sns

# 拟合图象的绘制
sns.set(color_codes=True)
sns.lmplot(x = 'sentiment', y = 'profit_per_month', data = sentiment_profit)


# In[28]:


sentiment_profit1=sentiment_profit[(sentiment_profit['sentiment']>0.4)&(sentiment_profit['profit_per_month']<30)]


# In[29]:


import pandas as pd 
import seaborn as sns

# 拟合图象的绘制
sns.set(color_codes=True)
model=sns.lmplot(x = 'sentiment', y = 'profit_per_month', data = sentiment_profit1)


# In[30]:


# Importing all the libraries we will use in this demo

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor # using package of testing VIF in statsmodels
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np


# In[31]:


# We try another way to build the regression model that without using the formula api
# Case sensitive for the names of dataframe

y = sentiment_profit1.profit_per_month
X = sentiment_profit1[['sentiment']].assign(const=1)

results = sm.OLS(y, X).fit()
print(results.summary())


# In[32]:


# We try another way to build the regression model that without using the formula api
# Case sensitive for the names of dataframe

y = reviews_details.sentiment
X = reviews_details[['host_response_time',
            'host_response_rate',
            'host_is_superhost',
            'host_has_profile_pic',
            'host_identity_verified']].assign(const=1)

results = sm.OLS(y, X).fit()
print(results.summary())


# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# ====热力图
from matplotlib.ticker import FormatStrFormatter
encoding="utf-8"
data = reviews_details     #读取数据
 
# 计算两两属性之间的皮尔森相关系数
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
 
# 返回按“列”降序排列的前n行
k = 30
cols = corrmat.nlargest(k, data.columns[0]).index
 
# 返回皮尔逊积矩相关系数
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt=".3f",
                 vmin=0,             #刻度阈值
                 vmax=1,
                 linewidths=.5,
                 cmap="RdPu",        #刻度颜色
                 annot_kws={"size": 10},
                 xticklabels=True,
                 yticklabels=True)             #seaborn.heatmap相关属性
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.ylabel(fontsize=15,)
# plt.xlabel(fontsize=15)
plt.title("主要变量之间的相关性强弱", fontsize=20)
plt.show()


# In[34]:


# create positive comments
df_positive = reviews_details[reviews_details['sentiment']>0]
df_positive[['comments','sentiment']].head()


# In[35]:


# create negative comments
df_negative = reviews_details[reviews_details['sentiment']<0]
df_negative[['comments','sentiment']].head()


# In[36]:


df_positive.columns


# In[37]:


# Importing libs
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Create the data
price=reviews_details['price_$']
sentiment=reviews_details['sentiment']

# # Create and shor the 2D Density plot
# ax = sns.kdeplot(price, sentiment, cmap="Blues", shade=False, bw=.15, cbar=True)
# ax.set(xlabel='price', ylabel='sentiment')
# plt.show()


# In[38]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import nltk


# In[39]:



def sentiment(rating):
  if (rating<1) & (rating>0):
    return 1
  else:
    return -1  
reviews_details['Sentiment_posneg'] = reviews_details['sentiment'].apply(sentiment)
reviews_details.head()


# In[40]:


fig = go.Figure([go.Bar(x=reviews_details.Sentiment_posneg.value_counts().index, y=reviews_details.Sentiment_posneg.value_counts().tolist())])
fig.update_layout(
    title="Values in each Sentiment",
    xaxis_title="Sentiment",
    yaxis_title="Values")
fig.show()


# ## Figure 3.8 box plots of positive and negative reviews sentiment

# In[41]:


# box plots of positive reviews sentiment

box = df_positive['sentiment']

plt.figure(figsize=(5,10))#设置画布的尺寸
plt.title('positive sentiment of boxplot',fontsize=20)#标题，并设定字号大小

#boxprops：color箱体边框色，facecolor箱体填充色；
plt.boxplot([box],patch_artist = True, boxprops = {'color':'orangered','facecolor':'pink'})

plt.show()#显示图像


# In[42]:


# box plots of negative reviews sentiment

box = df_negative['sentiment']

plt.figure(figsize=(5,10))#设置画布的尺寸
plt.title('negative sentiment of boxplot',fontsize=20)#标题，并设定字号大小

#boxprops：color箱体边框色，facecolor箱体填充色；
plt.boxplot([box],patch_artist = True, boxprops = {'color':'orangered','facecolor':'pink'})

plt.show()#显示图像


# ## Figure 3.9 scatter plot of positive and negative reviews

# In[43]:


import seaborn as sns 
ax = sns.scatterplot(data=reviews_details,x=reviews_details['sentiment'].index,y=reviews_details['sentiment'])
ax.set_ylabel('Sentiment') 
plt.show()


# In[44]:


#draw scatter plot of positive reviews
import seaborn as sns 
ax = sns.scatterplot(data=df_positive,x=df_positive['sentiment'].index,y=df_positive['sentiment'])
ax.set_ylabel('Positive Sentiment') 
plt.show()


# In[45]:


# #draw scatter plot of negative reviews
# import seaborn as sns
# sns.scatterplot(data=df_negative,x=df_negative['sentiment'].index,y=df_negative['sentiment'])

#draw scatter plot of positive reviews
import seaborn as sns 
ax = sns.scatterplot(data=df_negative,x=df_negative['sentiment'].index,y=df_negative['sentiment'])
ax.set_ylabel('Negative Sentiment') 
plt.show()


# In[46]:


#draw scatter plot of all cleansed reviews
import seaborn as sns
sns.scatterplot(data=reviews_details,x=reviews_details['sentiment'].index,y=reviews_details['sentiment'])

#draw scatter plot of positive reviews
import seaborn as sns
sns.scatterplot(data=df_positive,x=df_positive['sentiment'].index,y=df_positive['sentiment'])
#draw scatter plot of negative reviews
import seaborn as sns
sns.scatterplot(data=df_negative,x=df_negative['sentiment'].index,y=df_negative['sentiment'])


# In[47]:



# find the most used words using the CountVectorizer() function of sklearn.
texts = df_positive.comments.tolist()

vec = CountVectorizer().fit(texts)
bag_of_words = vec.transform(texts)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

cvec_df_positive = pd.DataFrame.from_records(words_freq, columns= ['words', 'counts']).sort_values(by="counts", ascending=False)


# In[48]:



# find the most used words using the CountVectorizer() function of sklearn.
texts = df_negative.comments.tolist()

vec = CountVectorizer().fit(texts)
bag_of_words = vec.transform(texts)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

cvec_df_negative = pd.DataFrame.from_records(words_freq, columns= ['words', 'counts']).sort_values(by="counts", ascending=False)




# In[50]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/22 20:58
from io import StringIO

import pandas as pd
import pydotplus
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from config import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,     roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re
from os import path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
import pandas as pd

warnings.filterwarnings('ignore')


# In[51]:


# 2. the most top 30 words in newreview data
# the reviews are mainly from the 'comment' column.

reviews_details = df_concat[
    ['id', 'name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments', 'review_scores_value']]

#
# host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
#     name="number_of_reviews")


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


# In[52]:


# 对comment列进行完整处理
reviews_details = col_comment_processing(reviews_details)


def comment_to_sentimentStrength(reviews_details):
    """
    通过对文本情感强度进行分析，获得对sentiment进行一个计算，增加了一个新的列，能够用于回归任务
    1. SentimentIntensityAnalyzer
        This will output the sentiment score of the text, with a value between -1 (very negative) and 1 (very positive).

    2. textblob
        This will output the sentiment score of the text, with a value between -1 (very negative) and 1 (very positive).

    新需求：如何构建离散化评价值？

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

# 上述的内容仍然属于特征工程的内容
# 用于构建完整数据集
# reviews_details.to_csv("./data/sentiment/reviews_details.csv", index=False)
# df_positive.to_csv("./data/sentiment/df_positive.csv", index=False)
# df_negative.to_csv("./data/sentiment/df_negative.csv", index=False)
# 如果需要使用完整数据集进行读取：


# In[53]:


reviews_details.to_csv('reviews_details.xls')


# In[54]:


# ---------------------------------------- 1. 针对文本情感分析分类----------------------
#
# 分析：其实从箱线图的分布可以看出来按照这个做分类的效果并不会很好
df_positive.boxplot(column="review_scores_value")
plt.show()

# 先选取正向文本进行测试
texts = df_positive.comments.tolist()
def tfidf_commnets_vector(texts):
    """
    文本向量化的提取方法
    由于构建完整版本的文本向量会消耗大量的计算资源，因此这里采用了部分采样。
    TfidfVectorizer 的效果就是CountVectorizer、TfidfTransformer的结合体
    :param texts:
    :return:
    """
    # tfidf 提取方式
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)
    # 构建用于文本情感分类的xy数据
    X = tfidf_vectorizer_vectors.toarray()
    return X

X = tfidf_commnets_vector(texts)
Y_clf = df_positive.review_scores_value


# ## Figure 3.11 key positive words TF-IDF value visualization

# In[55]:


# 输出top100 关键词
def top_words_importance_plt(texts):
    """
    使用tf-idf对输入文本内容进行处理
    输出top 100 的关键词 以及 对应的词云
    :param texts:
    :return:
    """
    from wordcloud import WordCloud, STOPWORDS
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
    features = vectorizer.get_feature_names_out()
    importances = vectorizer.idf_
    # Zip the features and importance weights together
    keywords = zip(features, importances)
    # Sort the keywords by importance weight in descending order
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    # Print the top 100 keywords and their importance weights
    for keyword, importance in keywords[:100]:
        print(f'{keyword}: {importance:.2f}')

     # construct the wordcloud
    # Extract the top 100 keywords and their importance weights
    keywords_cloud = [(keyword, importance) for keyword, importance in zip(features, importances)][:100]
    keywords_cloud = dict(keywords_cloud)

    stopwords = set(STOPWORDS)
    # img = imageio.imread('./data/alice_mask.png')

    # Create a wordcloud object
    wordcloud = WordCloud(max_words=2000,
                          background_color="white",  # 设置背景颜色
                          width=1000, height=860,
                          # mask=img,# 设置背景图片
                          stopwords=stopwords
                          )

    # Generate the wordcloud from the keywords and importance weights
    wordcloud.generate_from_frequencies(keywords_cloud)

    # Plot the wordcloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # plt.savefig('tfidf_top100_wordCloud.png",dpi=500,bbox_inches = 'tight') # save the image.

top_words_importance_plt(texts)


# In[56]:


# key positive words visualization
from matplotlib import pyplot as plt

positive_words = ['paul','parking','kitchen','family','transport',
                  'beds','modern','friendly','free wi-fi','pool',
                  'hosts','quick book','amenities','restaurants',
                  'super','spacious','convenient','clean'
                 ]
tfidf_values = [4.62,4.45,4.43,4.34,4.32,
          4.29,4.26,4.17,4.12,4.11,4.03,
          4.02,3.95,3.93,3.90,3.90,3.72,2.35
         ]

plt.figure()
plt.bar(positive_words, tfidf_values)
plt.ylabel("positive TFIDF-value")
plt.xticks(rotation=90) # 旋转90度
plt.show()


# In[57]:


# key negative words visualization
from matplotlib import pyplot as plt

negative_words = ['park','noise','photos','pool','towels',
                  'located','small','property','toilet','money',
                  'bedroom','beds','communication','kitchen,city',
                  'sleep','floor','clean','bathroom','night',
                  'room','dirty','host','location','apartment'
                 ]
tfidf_values = [4.33,3.99,3.96,3.96,3.96,
          3.92,3.92,3.89,3.83,3.80,
          3.77,3.77,3.77,3.77,3.61,
          3.54,3.50,3.41,3.30,3.18,
          3.17,2.91,2.60,2.43
         ]

plt.figure()
plt.bar(negative_words, tfidf_values)
plt.ylabel("negative TFIDF-value")
plt.xticks(rotation=90) # 旋转90度
plt.show()


# In[58]:


text=df_positive['comments']


# In[59]:


top_words_importance_plt(text)


# In[60]:


df_positive.head()


# In[63]:


df_reviews=df_concat


# In[68]:


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
df_reviews=df_review_processing(df_reviews)


# In[70]:




def sentiment_clf_rfc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # ----------------- 模型1 随机森林 -----------------
    # Create a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred_test = clf.predict(X_test)
    # Evaluate the performance of the classifier
    accuracy = clf.score(X_test, y_test)
    print('Accuracy of classification:', accuracy)

    cm = confusion_matrix(y_test, y_pred_test)
    # print('Confusion matrix\n', cm)
    cm_matrix = pd.DataFrame(data=cm
                             # columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'],
                             # index=['Predict Negative', 'Predict Neutral', 'Predict Positive']
                             )
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

    # 文本情感分析做分类，如果需要做回归的话用下面的代码，但是基本上都是做分类的
    # Y_clf = df_positive.review_scores_value[:1000]
    # X_train,X_test,y_train,y_test=train_test_split(X, Y_reg, test_size=0.2)
    # reg = RandomForestRegressor()
    # reg.fit(X_train, y_train)
    # accuracy = reg.score(X_test, y_test)
    # print('R2 of regression:', accuracy)


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


# 输出top100 关键词
# top_words_importance_plt(texts)

# 随机森林分类模型
# sentiment_clf_rfc(X, Y_clf)

# 决策树模型
# sentiment_clf_dt(X, Y_clf)

# 后面再增加几个分类模型和一个集成模型，来对比一下效果可以（不一定必要）


# -------------------------------- 2. 使用随机森林构建权重分析 -----------------------
# 这一部分内容按照所有特征列,全部需要进行转化，并且不会用到有关NLP部分的col，并将review_scores_value作为最终的分类目标
# 最终处理完成的数据集是reviews_details
# 数据集需要重新更换成，并且构建两个随机森林：
# 1.找到 average monthly sentiment最大的100家home（房源）做随机森林 name
# 2.找到 average profit per month 最大的100家host(房东）做随机森林 host_name
# 3.构建home（房源）当中所有月平均
# 4，构建所有host name当中的月平均
# 注意：
# 1. 为了能够完成处理上述使用的reviews_details是缩小版本的，因为做了文本情感分类。
#    所以这个地方的需要重新从df_concat 当中构建
# 2. 由于前面代码的混乱，导致数据问题，所以我在这里只实现必要的特征，需要添加之前处理的
#    特征请替换最开始读取的特征即可，然后在下面选择特征列的位置进行确认


scorer = SentimentIntensityAnalyzer()
def calculate_sentiment(comment):
    return (scorer.polarity_scores(comment)['compound'])
df_concat.loc[:, 'sentiment'] = df_concat['comments'].apply(calculate_sentiment)

# 构建特征
numerical_fea = list(df_concat.select_dtypes(exclude=['category']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(df_concat.columns)))
selected_cols = numerical_fea + [""]


# In[63]:



# -------------------------------- 构建Xy --------------------------
# TODO：将filtered_df1和filtered_df2 对下面的df进行替换就可以进行计算了
df = df_concat.select_dtypes(exclude='category') # 这个地方对列进行选择！
# df = filtered_df1.select_dtypes(exclude='category')
# df = filtered_df2.select_dtypes(exclude='category')
df = df.dropna()
X = df.drop('review_scores_value', axis=1)
y = df.review_scores_value[:X.shape[0]]  # 20000是最开始选择到的，28行


def rfc_process(X, y):
    """
    随机森铃实现分类，同时输出最终的权重比例
    opt会消耗大量计算资源，谨慎运行
    :param X:
    :param y:
    :return:
    """
    ros = RandomOverSampler()
    x_train, y_train = ros.fit_resample(X, y)

    train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.3)

    # ------------------------ model ------------------
    model = RandomForestClassifier()

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('reduce_dim', PCA()),
                     ('classifier', model)])
    pipe.fit(train_x, train_y)
    train_pred = pipe.predict(train_x)
    val_pred = pipe.predict(val_x)

    # ----------------------- find the importance tree -----------------
    dot_data = StringIO()
    export_graphviz(pipe.named_steps['classifier'].estimators_[0],
                    out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    Image(graph.create_png())

    # ----------------------- feature importance -----------------
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = x_train.columns

    for f in range(x_train.shape[1]):
        print("{}) {} is {}".format(f + 1, feat_labels[indices[f]], importances[indices[f]]))



    # ----------------------- opt -------------------
    #
    # n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=10)]
    # max_features = [x+1 for x in range(11)]
    # max_depth = [int(x) for x in np.linspace(start=1, stop=11, num=5)]
    # min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=5)]
    # params = {'n_estimators': n_estimators,
    #           'max_depth': max_depth,
    #           'max_features': max_features,
    #           'min_samples_split': min_samples_split
    #           }
    # cls = RandomizedSearchCV(model, params, cv=5, n_iter=10,n_jobs = -1)
    # cls.fit(train_x, train_y)
    # val_pred = cls.predict(val_x)
    # train_pred = cls.predict(train_x)
    #
    # best_estimator = cls.best_estimator_
    # print(best_estimator)
    # print(cls.best_score_)

    # ---------------------- model performance ----------------
    print('Training RMSE:{}'.format(np.sqrt(mean_squared_error(train_y, train_pred))))
    print('Test RMSE:{}'.format(np.sqrt(mean_squared_error(val_y, val_pred))))
    print('Training R-squared:{}'.format(r2_score(train_y, train_pred)))
    print('Test R-squared:{}'.format(r2_score(val_y, val_pred)))

rfc_process(X,y)

print("Success")

"""
1) id is 0.160980937415915
2) host_id is 0.062295402510985476
3) reviewer_id is 0.047857142517669864
4) review_scores_checkin is 0.038792651098771856
5) scrape_id is 0.03663752141938812
6) minimum_maximum_nights is 0.03243774857885271
7) number_of_reviews_ltm is 0.03101334922402329
8) minimum_minimum_nights is 0.030406201527290278
9) review_scores_rating is 0.03006037068845453
10) distance is 0.028389888378240306
11) host_response_rate is 0.02712361366341895
12) review_scores_avg6 is 0.02551392928024227
13) calculated_host_listings_count_private_rooms is 0.02432505325321922
14) reviews_per_month is 0.02430095892878475
15) review_scores_communication is 0.0238405721191318
16) review_scores_accuracy is 0.02358246053870515
17) year is 0.02304557155652446
18) calculated_host_listings_count_shared_rooms is 0.020677578082040425
19) review_scores_cleanliness is 0.019986240366508617
20) price_$ is 0.019959653763902695
21) month is 0.019076011272909445
22) profit_per_month is 0.018129883087404618
23) availability_365 is 0.01792290611867796
24) calculated_host_listings_count is 0.01680420059228854
25) availability_60 is 0.016382837764304235
26) maximum_maximum_nights is 0.016077600890112995
27) host_response_time is 0.015120599338200265
28) instant_bookable is 0.014659145191305767
29) number_of_reviews_l30d is 0.012914872669333135
30) review_scores_location is 0.012912611580254616
31) calculated_host_listings_count_entire_homes is 0.012495435578542718
32) host_is_superhost is 0.011989580905720908
33) availability_90 is 0.011641624952983529
34) minimum_nights_avg_ntm is 0.011119854526791172
35) number_of_reviews is 0.010059618904382927
36) maximum_minimum_nights is 0.009152643575809548
37) host_has_profile_pic is 0.008703581273673365
38) latitude is 0.007614653820360524
39) minimum_nights is 0.0067594991594722795
40) host_listings_count is 0.006465553978162195
41) maximum_nights_avg_ntm is 0.0036795463021276537
42) longitude is 0.0025930808319256844
43) maximum_nights is 0.0024062204345222564
44) host_identity_verified is 0.0017929402663335065
45) beds is 0.0011044383526854571
46) availability_30 is 0.000560437174477732
47) accommodates is 0.00048207441898995567
48) bedrooms is 0.00015170212617725572
49) day is 0.0
50) sentiment is 0.0

原始数据长这样
因此为了做多元回归，我们对数据进行了一个定义，必须是人的行为+商户的评价+

4) review_scores_checkin is 0.038792651098771856
6) minimum_maximum_nights is 0.03243774857885271
7) number_of_reviews_ltm is 0.03101334922402329
8) minimum_minimum_nights is 0.030406201527290278
9) review_scores_rating is 0.03006037068845453
10) distance is 0.028389888378240306
11) host_response_rate is 0.02712361366341895
12) review_scores_avg6 is 0.02551392928024227
13) calculated_host_listings_count_private_rooms is 0.02432505325321922
14) reviews_per_month is 0.02430095892878475
18) calculated_host_listings_count_shared_rooms is 0.020677578082040425
19) review_scores_cleanliness is 0.019986240366508617
20) price_$ is 0.019959653763902695
27) host_response_time is 0.015120599338200265


"""


# In[ ]:


get_ipython().system('pip install GraphViz')


# In[ ]:


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
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,     roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import re
import seaborn as sns
import matplotlib.pyplot as plt
#import lightgbm
import warnings
import pandas as pd
warnings.filterwarnings('ignore')



# 2. the most top 30 words in newreview data
# the reviews are mainly from the 'comment' column.

reviews_details = df_concat[
    ['id', 'name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments', 'review_scores_value']]

#
host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
    name="number_of_reviews")


def col_comment_processing(reviews_details):
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


reviews_details = col_comment_processing(reviews_details)

# 下面需要对sentiment进行一个计算，增加了一个新的列，能够用于回归任务
# 主要使用到的技术是

# 最终获得的数据集：处理完毕的reviews_details
scorer = SentimentIntensityAnalyzer()

def calculate_sentiment(comment):
    return (scorer.polarity_scores(comment)['compound'])

# TODO： 这部分是构建出来的完整数据集
reviews_details.loc[:, 'sentiment'] = reviews_details['comments'].apply(calculate_sentiment)
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

# 对文本进行词频统计
vectorizer = CountVectorizer()
vectorizer.fit_transform(texts)
top_words = vectorizer.get_feature_names_out()
print("Top words: {}".format(top_words[:100]))

X = tfidf_vectorizer_vectors.toarray()
Y_clf = df_positive.review_scores_value[:1000]

X_train, X_test, y_train, y_test = train_test_split(X, Y_clf, test_size=0.2)

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


# 多个不同的模型进行比对测试
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
category_fea = list(filter(lambda x: x not in numerical_fea,list(df_concat.columns)))
selected_cols = numerical_fea+[""]
# ------------------ 任务数据集构建 ---------------
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
y = df.review_scores_value[:20000] # 20000是最开始选择到的，28行



def rfc_process(X,y):
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


# In[ ]:


from io import StringIO
import pandas as pd
import pydotplus
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PowerTransformer
from config import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,     roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re
from os import path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
# import lightgbm
import warnings
import pandas as pd


# In[ ]:


def regression(df):
    """
    # Check the assumptions of linear regression
    # Check for linear relationship between features and target
    # Check for independence and normality of errors
    # Check for constant variance of errors
    :param X:
    :param y:
    :return:
    """
    # features = ["review_scores_checkin",
    #             "minimum_maximum_nights",
    #             "number_of_reviews_ltm",
    #             "minimum_minimum_nights",
    #             "review_scores_rating",
    #             "distance",
    #             "host_response_rate",
    #             "review_scores_avg6",
    #             "calculated_host_listings_count_private_rooms",
    #             "reviews_per_month",
    #             "calculated_host_listings_count_shared_rooms",
    #             "review_scores_cleanliness",
    #             "price_$",
    #             "host_response_time"]



    # X = X[features]
    # X=df.astype("float64")
    # X = df_concat[['host_response_time',
    #                      'host_response_rate',
    #                      'host_is_superhost',
    #                      'host_has_profile_pic',
    #                      'host_identity_verified']].assign(const=1)
    data = df_concat.select_dtypes(exclude='category')
    X = data.astype("float64")
    X = X.astype("float64")
    y = df_concat.sentiment + 1

    # Choose the right features
    # Use recursive feature elimination (RFE) to select the most important features
    selector = RFE(LinearRegression())
    selector = selector.fit(X, y)
    X_new = X[X.columns[selector.support_]]

    # Check for multicollinearity
    # Compute variance inflation factor (VIF) for each feature
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]

    # Remove features with high VIF
    threshold = 5
    res = []
    for i in vif:
        res.append(i > threshold)
    X_new = X_new.drop(X_new.columns[res], axis=1)

    # Transform the target variable
    # Use Box-Cox transformation
    from scipy.stats import boxcox
    y_transformed, _ = boxcox(y)
    ## Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X_new, y_transformed, test_size=0.2)
    #
    # # Create a linear regression model
    # model = LinearRegression()
    # # Fit the model to the training data
    # model.fit(X_train, y_train)
    # # Predict the response for the testing data
    # y_pred = model.predict(X_test)
    #
    # # Evaluate the model's performance
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(f'MAE: {mae:.2f}')
    # print(f'R2: {r2:.2f}')
    # # Print the model's coefficients
    # print(f'Coefficients: {model.coef_}')

    import statsmodels.api as sm

    results = sm.OLS(y, X).fit()
    print(results.summary())

    # 绘制热力图
    import seaborn as sns
    # Create a larger figure
    plt.figure(figsize=(20, 18))
    data = X_new.corr()
    # Draw the heat map
    sns.heatmap(data, cmap='cool', annot=True, fmt='.1f')
    # Show the plot
    plt.show()
    plt.savefig("./data/hotmap.jpg",dpi=500,bbox_inches = 'tight')

regression(df)


# In[ ]:


cvec_df_positive.head(100)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
# just send in all your docs here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)


# In[ ]:


X = tfidf_vectorizer_vectors.toarray()
Y = df_positive['sentiment'][2:1843]


# In[ ]:


X.shape


# In[ ]:


Y.shape


# In[ ]:


Y.head()


# In[ ]:


import plotly.graph_objects as go


# In[ ]:


fig = go.Figure([go.Bar(x=Y.value_counts().index, y=Y.value_counts().tolist())])
fig.update_layout(
    title="Values in each Sentiment",
    xaxis_title="Sentiment",
    yaxis_title="Values")
fig.show()


# In[ ]:


## positive reviews
# 柱状图展示
# cvec_df.head(10).plot.barh(figsize=(15, 3), width=0.4)
plt.barh(cvec_df_positive.head(20).words.to_list(),
         cvec_df_positive.head(20).counts.to_list()
         )
plt.show()

# 词云展示
cvec_dict = dict(zip(cvec_df_positive.words, cvec_df_positive.counts))

wordcloud = WordCloud(width=800, height=400)
wordcloud.generate_from_frequencies(frequencies=cvec_dict)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


get_ipython().system('pip install jieba')


# In[ ]:


# 提取某列保存为txt文件
df_positive['comments'].to_csv("column.txt", index=False, header=False)


# In[ ]:


import jieba
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# 词云展示
cvec_dict = dict(zip(cvec_df_positive.words, cvec_df_positive.counts))

wordcloud = WordCloud(width=800, height=400)
wordcloud.generate_from_frequencies(frequencies=cvec_dict)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

def trans_ch(txt):
    words = jieba.lcut(txt)
    newtxt = ''.join(words)
    return newtxt


txt = pd.read_csv("column.txt")
mask = np.array(Image.open("airbnb_wordcloud1.png"))
wordcloud = WordCloud(background_color="red",
                      width=800,
                      height=600,
                      max_words=200,
                      max_font_size=80,
                      mask=mask,
                      contour_width=4,
                      contour_color='steelblue',
                      font_path="msyh.ttf"
                      )

wordcloud.generate(txt)

# 生成的词云图像保存到本地
wordcloud.to_file('positive_airbnb_wordcloud.png')


# In[ ]:


## negative reviews
# 柱状图展示
# cvec_df.head(10).plot.barh(figsize=(15, 3), width=0.4)
plt.barh(cvec_df_negative.head(20).words.to_list(),
         cvec_df_negative.head(20).counts.to_list()
         )
plt.show()

# 词云展示
cvec_dict = dict(zip(cvec_df_negative.words, cvec_df_negative.counts))

wordcloud = WordCloud(width=800, height=400)
wordcloud.generate_from_frequencies(frequencies=cvec_dict)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


cvec_df_negative.head(100)


# In[ ]:


# Statistics
import pandas as pd
import numpy as np
import math as mt

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# In[ ]:


#drop rows that 'listing_id' doesn't exist in the 'id' of 'listings' and create a new dataset'concat'
# merge dataset 'listings' and 'reviews' with the same 'id'

df_positive = pd.merge(df_concat,df_positive,how="inner",on='id')


# In[ ]:


df_positive.head()


# In[ ]:





# In[ ]:





# In[ ]:




