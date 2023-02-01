import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from config import reduce_mem_usage


def col_comment_processing(reviews_details):
    """
    完整数据集，针对评论数据当中的comments 列进行处理
    要求：
    1. 实现commnets文本评论的清洗
    2. 仅保留comments当中的英文评论
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


# 输出top100 关键词
def top_words_importance_plt(texts):
    """
    实现内容
    1. 使用tf-idf对输入文本内容进行处理
    2. 输出top 100 的关键词 以及 对应的词云(如果要改变词云形状需要自定义图片和输入路径)
    3. LDA模型
    4. GSDMM模型
    :param texts:
    :return:
    """
    import imageio
    from wordcloud import WordCloud, STOPWORDS

    # TODO: 对文本进行词频统计
    # vectorizer = CountVectorizer()
    # vectorizer.fit_transform(texts)
    # top_words = vectorizer.get_feature_names()
    # print("Top words: {}".format(top_words[:100]))

    # TODO:tfidf
    # TfidfVectorizer的效果就是CountVectorizer、TfidfTransformer的结合体, the larger data, the higher performance.
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    # Fit the vectorizer to the texts
    vectorizer.fit(texts)

    # Step one :Extract the top features and their importance weights
    features = vectorizer.get_feature_names()
    importances = vectorizer.idf_

    # ------------------------------ lda -------------------------------
    """
    taks:
    1. LDA with pyLDAvis for visualization
    2. LDA can do high performance in large texts(more than 50 words)
    但在尝试对较短文本的主题进行建模时，它的性能往往会急剧下降，
    原因很明显，短文本（如推特或论坛问题的标题）可能只涉及单个主题
    
    """

    X = vectorizer.fit_transform(texts)

    # ----- opt to determine the n_components ---------
    # Define a grid of values for n_components
    param_grid = {'n_components': [1,2, 3, 4]}
    # Create a LatentDirichletAllocation object
    lda = LatentDirichletAllocation()
    # Use GridSearchCV to find the optimal value of n_components
    model = GridSearchCV(lda, param_grid, cv=5)
    model.fit(X)
    # Print the optimal value of n_components
    print(model.best_params_) # {'n_components': 1}


    n_topics = 1 # 指定 lda 主题数根据困惑度和一致性判断最佳主题数
    # 要输出的每个主题的前 n_top_words 个主题词数
    n_top_words = 30
    # use the best paramaters.
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=50,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    # 核心，给 LDA 生成的 TF-IDF 矩阵
    lda.fit(X)


    # TODO:visualization 2
    # Get the topic-word distributions from the LDA model
    topic_word = lda.components_

    # Get the top 100 key words for each topic
    key_words = []
    for topic in topic_word:
        word_idx = np.argsort(topic)[::-1][:100]
        key_words.append([features[i] for i in word_idx])

    # Concatenate the key words into a single list
    key_words = [word for topic in key_words for word in topic]

    # Generate the word cloud
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_words=2000,
                          background_color="white",  # 设置背景颜色
                          width=1000, height=860,
                          # mask=img,# 设置背景图片
                          stopwords=stopwords
                          ).generate(' '.join(key_words))

    # Display the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # TODO: 可视化方法3
    # Get the topic assignments for each document
    topics = lda.transform(X)
    # Get the top words and sentences for each topic
    num_top_words = 10
    num_top_sentences = 5
    for topic_idx, topic in enumerate(lda.components_):
        print(f'Top words for topic {topic_idx}:')
        top_words = [features[i] for i in topic.argsort()[::-1][:num_top_words]]
        print(top_words)
        print()
        print(f'Top sentences for topic {topic_idx}:')
        top_sentences = np.argsort(topics[:, topic_idx])[::-1][:num_top_sentences]
        for i in top_sentences:
            print(texts[i])
        print()

    # TODO：visualization 1
    import pyLDAvis.sklearn
    # Prepare the data for visualization
    vis_data = pyLDAvis.sklearn.prepare(lda_model = lda,
                                        dtm=X,
                                        vectorizer=vectorizer,)
    # Display the visualization
    # pyLDAvis.display(vis_data) # notebook display
    pyLDAvis.show(vis_data, open_browser=False, local=False) # web


    # ------------- GSDMM ------------------
    # that is an Algorithm from paper ?


    def wordcloud_with_topdict(features, importances):
        """

        :param features:
        :param importances:
        :return:
        """
        # --------------------------- visualization ------------------
        # TODO: for showing the top 100.
        # 这个需要重要程度权重构建
        # Zip the features and importance weights together
        keywords = zip(features, importances)
        # Sort the keywords by importance weight in descending order
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        # Print the top 100 keywords and their importance weights
        for keyword, importance in keywords[:100]:
            print(f'{keyword}: {importance:.2f}')

        # TODO: Construct the wordcloud
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
        wordcloud.generate_from_frequencies(keywords_cloud) # 在这个地方进行文本替换即可

        # Plot the wordcloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        # plt.savefig('tfidf_top100_wordCloud.png",dpi=500,bbox_inches = 'tight') # save the image.



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
    cm_matrix = pd.DataFrame(data=cm,
                             # columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'],
                             # index=['Predict Negative', 'Predict Neutral', 'Predict Positive']
                             columns=['Actual Negative', 'Actual Positive'],
                             index=['Predict Negative', 'Predict Positive']
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
    cm_matrix = pd.DataFrame(data=cm,
                             # columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'],
                             # index=['Predict Negative', 'Predict Neutral', 'Predict Positive']
                             columns=['Actual Negative', 'Actual Positive'],
                             index=['Predict Negative', 'Predict Positive']
                             )
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

def sentiment_TO_cls(reviews_details):
    def sentiment(rating):
        if 0.0 <= rating and rating < 0.2:
            return 1
        elif 0.2 <= rating and rating < 0.4:
            return 2
        elif 0.4 <= rating and rating < 0.6:
            return 3
        elif 0.6 <= rating and rating < 0.8:
            return 4
        elif 0.8 <= rating and rating < 1.0:
            return 5
        else:
            return 0
        # else:
        #     return 0

    # reviews_details['Sentiment_posneg'] = reviews_details['sentiment'].apply(sentiment)
    reviews_details['sentiment'] = reviews_details['sentiment'].apply(sentiment)

    return reviews_details

def reformat_dataset():
    """
    template. Not used yet.
    :return:
    """
    # ---------------------------------------- 1. 针对文本情感分析分类----------------------
    # df_concat = pd.read_csv('./data/clean/df_concat.csv', nrows=20000)
    df_concat = pd.read_csv('./data/clean/df_concat.csv')
    df_concat = reduce_mem_usage(df_concat)

    # 2. the most top 30 words in newreview data
    # the reviews are mainly from the 'comment' column.

    # reviews_details = df_concat[
    #     ['id', 'name', 'host_id', 'host_name', 'date', 'reviewer_id', 'reviewer_name', 'comments',
    #      'review_scores_value']]
    reviews_details = df_concat

    #
    # host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(
    #     name="number_of_reviews")

    # TODO:构建情感分析数据
    # 对comment列进行完整处理
    reviews_details = col_comment_processing(reviews_details)
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

    # 分析：其实从箱线图的分布可以看出来按照这个做分类的效果并不会很好
    # df_positive.boxplot(column="review_scores_value")
    # plt.show()



if __name__ == '__main__':
    # TODO: 构建完毕之后直接读取就行了
    reviews_details = pd.read_csv("./data/seasonal/quarter_winter.csv")
    reviews_details = reduce_mem_usage(reviews_details)

    # TODO:sentiment离散化
    reviews_details = sentiment_TO_cls(reviews_details) # 注意在这个地方会将sentimen离散化

    # TODO：文本向量化
    # 先选取正向文本进行测试
    texts = reviews_details.comments.tolist()[:5000]  # 积极评论

    # TODO: 模型效果评价
    # # step one: output the top 100 keywords.
    # tfidf+lda+gsdmm1
    top_words_importance_plt(texts)


    # ------------------------- classification model judgement -------------------
    # X = tfidf_commnets_vector(texts[:2000]) # 文本向量很大，因此只取部分来进行运算
    # Y_clf = reviews_details.sentiment[:2000] # 需要先将sentiment离散化
    # # random forest classification model
    # sentiment_clf_rfc(X, Y_clf)
    #
    # # decision tree classification model
    # sentiment_clf_dt(X, Y_clf)

    # To be continued:
    # Added a few clf or a esemble model to compare the performance.
