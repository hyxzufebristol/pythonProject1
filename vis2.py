# -*- coding: utf-8 -*- 
# @Time : 2022/12/19 10:17 
# @Author : YeMeng 
# @File : vis2.py 
# @contact: 876720687@qq.com
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from folium.plugins import FastMarkerCluster
from haversine import haversine
from config import *
from demo_Text_Sentiment_Analysis import col_comment_processing



# ---------------------------- 可视化展示 ----------------------------
def map_distance(df):
    """
    通过地图展示实际坐标位置
    :param listings:
    :return:
    """
    # the place that we want to go.
    destination = [-37.82, 144.96]
    def cal_distance(la, lon):
        loc2 = [la, lon]
        return haversine(destination, loc2)

    df['distance'] = df[['latitude', "longitude"]].apply(
        lambda x: cal_distance(x['latitude'], x['longitude'])
        , axis=1)
    # draw maps
    lats = df['latitude'].tolist()
    lons = df['longitude'].tolist()
    locations = list(zip(lats, lons))
    import folium

    map1 = folium.Map(location=destination, zoom_start=11.5)
    folium.Circle(radius=5000, location=destination, color="blue", fill=False).add_to(map1)
    # FastMarkerCluster(data=locations).add_to(map1)
    df_loc = df.loc[:["latitude", "longitude"]].values.tolist()
    marker_cluster = FastMarkerCluster(data=locations).add_to(map1)
    for loc in df_loc:
        folium.Marker(location=loc, icon=None).add_to(marker_cluster)
    # 输出图片
    # map1


def quick_plot(df):
    """
    这里用了快速展示的一个输出方法
    :return:
    """
    freq = df['room_type'].value_counts().sort_values(ascending=True)
    freq.plot.barh(figsize=(15, 3), width=0.7, color=["g", "b", "r", "y"])
    plt.show()

    # top host
    rank_data = df.groupby(['host_name'])['profit_per_month', 'reviews_per_month'].apply(sum)
    rank_data = rank_data.sort_values('profit_per_month', ascending=False)
    rank_data.iloc[:10, :1].plot.barh(figsize=(15, 3), width=0.4)
    plt.show()


def review_score_bar_plot(df):
    """
    对
    :return:
    """
    # review 一系列都是得分能够直接获得消费者对这家店的评价
    review_score = df[df['number_of_reviews'] >= 10]

    fig, ax = plt.figure(figsize=(20, 15))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    ax1 = fig.add_subplot(321)
    feq = review_score['review_scores_location'].value_counts().sort_index()
    ax1 = feq.plot.bar(color='b', width=0.7, rot=0)
    # ax1.tick_params(axis = 'both', labelsize = 16)
    plt.title("Location", fontsize=24)
    plt.ylabel('Number of listings', fontsize=14)
    plt.xlabel('Average review score', fontsize=14)

    ax2 = fig.add_subplot(322)
    feq = review_score['review_scores_cleanliness'].value_counts().sort_index()
    ax2 = feq.plot.bar(color='b', width=0.7, rot=0)
    plt.title("Cleanliness", fontsize=24)
    plt.ylabel('Number of listings', fontsize=14)
    plt.xlabel('Average review score', fontsize=14)

    ax3 = fig.add_subplot(323)
    feq = review_score['review_scores_value'].value_counts().sort_index()
    ax3 = feq.plot.bar(color='b', width=0.7, rot=0)
    plt.title("Value", fontsize=24)
    plt.ylabel('Number of listings', fontsize=14)
    plt.xlabel('Average review score', fontsize=14)

    ax4 = fig.add_subplot(324)
    feq = review_score['review_scores_communication'].value_counts().sort_index()
    ax4 = feq.plot.bar(color='b', width=0.7, rot=0)
    plt.title("Communication", fontsize=24)
    plt.ylabel('Number of listings', fontsize=14)
    plt.xlabel('Average review score', fontsize=14)

    ax5 = fig.add_subplot(325)
    feq = review_score['review_scores_checkin'].value_counts().sort_index()
    ax5 = feq.plot.bar(color='b', width=0.7, rot=0)
    plt.title("Arrival", fontsize=24)
    plt.ylabel('Number of listings', fontsize=14)
    plt.xlabel('Average review score', fontsize=14)

    ax6 = fig.add_subplot(326)
    feq = review_score['review_scores_accuracy'].value_counts().sort_index()
    ax6 = feq.plot.bar(color='b', width=0.7, rot=0)
    plt.title("Accuracy", fontsize=24)
    plt.ylabel('Number of listings', fontsize=14)
    plt.xlabel('Average review score', fontsize=14)

    plt.tight_layout()
    plt.show()




def comments_vercor_plot(df):
    # ------------------ 文本向量化的提取方法 ---------------------

    # find the most used words using the CountVectorizer() function of sklearn.
    # texts = reviews_details.comments.tolist()
    texts = df.comments.tolist()

    # 提取方法1
    texts = texts[:1000] # 减少计算量
    from sklearn.feature_extraction.text import TfidfVectorizer
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(texts)



    ## 方法2
    ## 这个计算方法保留待定
    vec = CountVectorizer().fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    cvec_df = pd.DataFrame.from_records(words_freq, columns=['words', 'counts']).sort_values(by="counts", ascending=False)

    # # 柱状图展示
    # cvec_df.head(10).plot.barh(figsize=(15, 3), width=0.4)
    # plt.barh(cvec_df.head(10).words.to_list(),
    #          cvec_df.head(10).counts.to_list()
    #          )
    # plt.show()

    # 词云展示
    cvec_dict = dict(zip(cvec_df.words, cvec_df.counts))

    wordcloud = WordCloud(width=800, height=400)
    wordcloud.generate_from_frequencies(frequencies=cvec_dict)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def Perimeter_number_listings_per_host(df):
    # 3. Perimeter of number of listings per host.
    # 我们需要找到最终入围的host有哪几个人？这个可以说是爱彼迎的优质会员了


    # host_names = df[['host_name']].drop_duplicates()
    # host_names = host_names.host_name.str.lower().tolist()
    # print(len(host_names))

    # 你需要什么内容这里选择一下就能显示了
    # col_describe = ["host_is_superhost","host_listings_count","accommodates","bedrooms","beds"]# 太多了，需要自己选
    # df.groupby(by="host_name")[col_describe].sum().describe()
    df.groupby(by="host_name").sum().describe()



def monthly_roomtype(df):
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




def show_monthly_price(df):
    """

    :param df:
    :return:
    """
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



def monthly_roomtype_pric(df):
    """

    :param df:
    :return:
    """
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



if __name__ == '__main__':
    df = pd.read_csv("./data/clean/df_concat.csv", nrows=10000)
    df = reduce_mem_usage(df)

    # 2. the most top 30 words in newreview data.
    # the reviews are mainly from the 'comment' column.
    reviews_details = df[['name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments']]

    host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
        name="number_of_reviews")

    # 通过第三部分的处理导入函数对comment列进行处理
    reviews_details = col_comment_processing(reviews_details)

    print(reviews_details.comments.values[2])

