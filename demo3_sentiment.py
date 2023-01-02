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
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score, \
    roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re
from os import path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd

from demo_Text_Sentiment_Analysis import sentiment_TO_cls

warnings.filterwarnings('ignore')


def construct_dataset(df_concat):
    """
    依据需求构建对应的数据集用于后续的分析。
    :param df_concat:
    :return:
    """
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

    # 针对任务3
    # df_concat.date = df_concat.date.astype("datetime64")
    # df_grouped = df_concat.groupby('host_name')
    # result = {}
    # output_sum = df_concat.groupby(['host_name'])['profit_per_month'].sum()
    # # df_concat[df_concat.host_name=="Abbie"]['date']
    # # df_grouped.date.groups["Abbie"]
    # # df_concat.iloc[7153].date

    # 构建数据集：


def rfc_process(df):
    """
    1. 随机森铃实现分类，同时输出最终的权重比例
    2. opt会消耗大量计算资源，谨慎运行
    3. 实现了对随机森林的可视化，会自动保存到文件目录
    :param X:
    :param y:
    :return:
    """
    X = df.drop('review_scores_value', axis=1)
    y = df.review_scores_value[:X.shape[0]]

    ros = RandomOverSampler()
    x_train, y_train = ros.fit_resample(X, y)

    train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.3)

    # ------------------------ model ------------------
    model = RandomForestClassifier()  # RandomForestRegressor、RandomForestClassifier
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('reduce_dim', PCA()),
                     ('classifier', model)])
    pipe.fit(train_x, train_y)
    train_pred = pipe.predict(train_x)
    val_pred = pipe.predict(val_x)

    # ----------------------- find the importance tree -----------------
    # 方法1
    # dot_data = StringIO()
    # export_graphviz(pipe.named_steps['classifier'].estimators_[0],
    #                 out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('tree.png')
    ## Image(graph.create_png())
    # 方法2
    # Iterate over the trees in the classifier
    from sklearn.tree import export_graphviz
    from graphviz import Source
    clf = pipe.steps[2][1]

    # Get the tree
    tree = clf.estimators_[-1]
    # Export the tree to a graphviz file
    export_graphviz(tree, out_file='tree.dot', feature_names=X.columns)
    # Visualize the tree
    with open('tree.dot') as f:
        dot_graph = f.read()
    s = Source(dot_graph)
    s.view()

    # ----------------------- feature importance -----------------
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = x_train.columns
    for f in range(x_train.shape[1]):
        print("{}) {} is {}".format(f + 1, feat_labels[indices[f]], importances[indices[f]]))

    # ----------------------- opt -------------------
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
    # best_estimator = cls.best_estimator_
    # print(best_estimator)
    # print(cls.best_score_)

    # ---------------------- model performance ----------------
    print('Training RMSE:{}'.format(np.sqrt(mean_squared_error(train_y, train_pred))))
    print('Test RMSE:{}'.format(np.sqrt(mean_squared_error(val_y, val_pred))))
    print('Training R-squared:{}'.format(r2_score(train_y, train_pred)))
    print('Test R-squared:{}'.format(r2_score(val_y, val_pred)))


def rfr_process(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X = df.drop('sentiment', axis=1)
    y = df.sentiment[:X.shape[0]]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    for f in range(X_train.shape[1]):
        print("{}) {} is {}".format(f + 1, feat_labels[indices[f]], importances[indices[f]]))

    # Create a pandas DataFrame with the feature names and importance scores
    df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

    # Sort the DataFrame by importance in descending order
    df.sort_values(by='importance', ascending=False, inplace=True)
    df = df.head(10)

    # plot a color histogram
    # Plot a color histogram of the feature importances
    # Choose a color palette
    colors = ['coral', 'dodgerblue', 'seagreen', 'orchid', 'darkorange', 'slategray', 'teal', 'purple', 'pink',
              'limegreen']

    # Plot a bar plot of the top 10 feature importances
    ax=df.plot.bar(x='feature', y='importance', color=colors, figsize=(8, 6))
    # Add text labels on top of the bars
    for i, label in enumerate(df['importance']):
        ax.text(i, label + 0.01, f'{label:.2f}', ha='center', va='bottom', fontsize=11)
    plt.title('Feature Importances')
    plt.ylabel('Importance Score')
    plt.xlabel('Feature')
    plt.tight_layout()
    # plt.xticks(rotation=45)
    plt.show()


def heatmap_demo(data):
    """
    输入数据就是完整的需要进行热力图绘制的数据集
    :param data:
    :return:
    """
    # --------------- 绘制热力图 ------------------
    import seaborn as sns
    # Create a larger figure
    plt.figure(figsize=(20, 18))
    # X = df_concat[['host_response_time',  # 可以通过这样的方式对需要的数据变量进行筛选
    #                'host_response_rate',
    #                'host_is_superhost',
    #                'host_has_profile_pic',
    #                'host_identity_verified',
    #                'sentiment']]
    # data = X
    # data['sentiment'] = y
    data = data.corr()
    # Draw the heat map
    ax = sns.heatmap(data, cmap='cool', annot=True, fmt='.3f',
                     annot_kws={"fontsize": 12})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    # Show the plot
    plt.title("Heat map of positive sentiment and amenities",
              fontdict={'weight': 'normal', 'size': 30})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    # plt.savefig("./data/hotmap.jpg",dpi=500,bbox_inches = 'tight')

def regression(df_concat):
    """
    # Check the assumptions of linear regression
    # Check for linear relationship between features and target
    # Check for independence and normality of errors
    # Check for constant variance of errors
    :param X:
    :param y:
    :return:
    """
    # ----------------------- 数据读取部分 --------------------
    # X = X[features]
    # X=df.astype("float64")
    X = df_concat[['host_response_time',  # 可以通过这样的方式对需要的数据变量进行筛选
                   'host_response_rate',
                   'host_is_superhost',
                   'host_has_profile_pic',
                   'host_identity_verified']]

    # data = df_concat.select_dtypes(exclude='category')
    # X = data.astype("float64")
    X.host_response_rate = X.host_response_rate.astype("float64")
    y = df_concat.sentiment + 1

    # ------------------------ 特征选择、消除共线性 ------------------------
    # TODO:
    # X=X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # X=X.dropna(axis=0)
    # TODO:Choose the right features
    # Use recursive feature elimination (RFE) to select the most important features
    # selector = RFE(LinearRegression())
    # selector = selector.fit(X, y)
    # X_new = X[X.columns[selector.support_]]
    X_new = X

    # TODO:Check for multicollinearity
    # Compute variance inflation factor (VIF) for each feature
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]

    # Remove features with high VIF
    threshold = 10  # not sure the suitable number
    res = []
    for i in vif:
        res.append(i > threshold)
    X_new = X_new.drop(X_new.columns[res], axis=1)

    # Transform the target variable
    # Use Box-Cox transformation
    from scipy.stats import boxcox
    y_transformed, _ = boxcox(y)

    def linearRegression_process(X_new, y_transformed):
        """
        使用正常线性回归,输入的数据是经过降维之后的
        :param X_new:
        :param y_transformed:
        :return:
        """

        ## Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_transformed, test_size=0.2)

        # Create a linear regression model
        model = LinearRegression()
        # Fit the model to the training data
        model.fit(X_train, y_train)
        # Predict the response for the testing data
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'MAE: {mae:.2f}')
        print(f'R2: {r2:.2f}')
        # Print the model's coefficients
        print(f'Coefficients: {model.coef_}')

    # linearRegression_process(X_new,y_transformed)

    import statsmodels.api as sm
    results = sm.OLS(y_transformed, X_new).fit()
    print(results.summary())


    # 绘制热力图
    X['sentiment']=y
    heatmap_demo(X)

def amenity_analysis(df):
    # 这是category里面嵌套的list
    # Use the get_dummies function to split the list column into multiple columns and convert the values to 01 values
    from sklearn.preprocessing import MultiLabelBinarizer
    import json

    df["amenities_list"] = df["amenities"].apply(lambda x: json.loads(x))
    # Create an instance of MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    # Fit the binarizer to the labels
    amenities_list_binary = mlb.fit_transform(df["amenities_list"])
    # X_data = pd.DataFrame(amenities_list_binary, columns=mlb.classes_, index=df.id)
    X_data = pd.DataFrame(amenities_list_binary, columns=mlb.classes_)

    df = sentiment_TO_cls(df)  # 将sentiment离散化
    df = df.reset_index()

    X_data['sentiment'] = df['sentiment']

    corr_data = X_data.corr().loc["sentiment"]
    corr_data[corr_data > 0].sort_values(ascending=False)
    data = X_data[corr_data[corr_data > 0].sort_values(ascending=False)[:20].index]
    heatmap_demo(data)  # plot


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


"""
There are several ways you can try to improve the R2 score of a linear regression model if it is not performing well. Here are a few suggestions:
Check the assumptions of linear regression: Linear regression assumes that the relationship between the features and the target is linear, that the errors are independent and normally distributed, and that the errors have constant variance. If any of these assumptions is not met, the model's performance may be poor.
Choose the right features: Linear regression is sensitive to the choice of features. Make sure that the features you select are relevant and have a strong relationship with the target. You can use techniques like feature selection or feature engineering to identify the most important features.
Check for multicollinearity: If the features are highly correlated with each other, it can affect the model's performance. You can check for multicollinearity using techniques like variance inflation factor (VIF) or by examining the correlation matrix.
Transform the target variable: If the target variable is not normally distributed, you can try transforming it using techniques like log transformation or Box-Cox transformation.
Try different model types: If linear regression is not performing well, you can try other types of models, such as non-linear models or ensemble models, which may be more suitable for your data.
"""
"""
文字说明：
多元线性回归流程
（前面用到的那个步骤，消除共线性RFE）
1. 首先我们使用函数rfr_process() 来构建随机森林pipline，其中主要用到的方法流程为：
    使用StandardScaler来进行归一化，然后使用PCA降维，能够降低模型变量的维度，然后使用随机森林的回归模型来进行拟合
    由于随机森林作为树状模型，对于多重共线性的敏感程度要比线性模型要低很多，因此我们使用归一化和PCA降维的方式来提升模型的优度，另外还使用Hyperparameter optimization method to improve model performance。
    另外，随机森林能够对变量进行重要程度分析，这部分内容有利于进一步选择和降低模型的维度，为我们构建线性回归模型提供一定的参考价值。
    并且我们输出了最终构建成功的模型的参数，可视化树状图，以及变量的重要程度。
2. 其次，使用regression()函数来构建回归模型：
    我们的最终目的是能够获得R2接近1并且MSE尽可能小的模型。
    值得注意的是在归回模型当中我们需要解决多重共线性问题，在本文当中使用了如下步骤进行模型的拟合：
    1. Choose the right features, Use recursive feature elimination (RFE) to select the most important features
    2. Check for multicollinearity, Compute variance inflation factor (VIF) for each feature
    3. 使用sm.OLS来对数据模型进行拟合，模型的最终结果（你自己描述一下）并使用热力图对数据变量的相关系数进行了可视化展示。
    
"""

if __name__ == '__main__':
    # df_concat = pd.read_csv('./data/clean/df_concat.csv', nrows=20000) # 防止计算量过大，没有进行全量运算
    # df_concat = pd.read_csv('./data/clean/df_concat.csv')
    df_concat = pd.read_csv('./data/sentiment/df_positive.csv')
    # df_concat = pd.read_csv('./data/sentiment/df_negative.csv')
    df_concat = reduce_mem_usage(df_concat)
    df = df_concat
    # TODO:构建特征
    # numerical_fea = list(df_concat.select_dtypes(exclude=['category']).columns)
    # category_fea = list(filter(lambda x: x not in numerical_fea, list(df_concat.columns)))
    # selected_cols = numerical_fea + [""]

    # # -------------------------------- 构建数据集 --------------------------
    # # X,y 在进入函数之后再进行拆分
    # # TODO：将filtered_df1和filtered_df2 对下面的df进行替换就可以进行计算了
    # # df = construct_dataset(df_concat) # 在这个地方进行数据集切换
    # df = df_concat.select_dtypes(exclude='category')  # 这个地方对列进行选择！通过上一个todo里面的selected_cols来进行选择
    # drops = [
    #     "id",
    #     "scrape_id",
    #     "host_id",
    #     "reviewer_id",
    #     "year",
    #     # "month",
    #     "review_scores_rating",
    #     "profit_per_month",
    #     "day",
    #     "is_english",
    #     "latitude",
    #     "longitude",
    #     "minimum_nights", "maximum_nights", "minimum_minimum_nights", "maximum_minimum_nights"
    #     , "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm"
    # ]
    # df = df.drop(columns=drops)
    # df = df.dropna()
    # # TODO:上述文本解释当中提及到的两个函数
    # #
    # # rfc_process(df)
    #
    # # regression(df)
    # print("Success")


    # # TODO:按照季节拆分成四个数据集
    # # 正常北半球：
    # # # Create a new column called "quarter"
    # df['quarter'] = (df['month'] - 1) // 3 + 1
    #
    # # Create four separate DataFrames for each quarter
    # quarter_1 = df[df['quarter'] == 1]
    # quarter_2 = df[df['quarter'] == 2]
    # quarter_3 = df[df['quarter'] == 3]
    # quarter_4 = df[df['quarter'] == 4]

    # 澳洲南半球：
    quarter_summer = df[(df['month'] == 1)|(df['month'] == 2)|(df['month'] == 12)]
    quarter_autumn = df[(df['month'] == 3)|(df['month'] == 4)|(df['month'] == 5)]
    quarter_winter = df[(df['month'] == 6)|(df['month'] == 7)|(df['month'] == 8)]
    quarter_spring = df[(df['month'] == 9)|(df['month'] == 10)|(df['month'] == 11)]
    # quarter_spring.to_csv("./data/seasonal/quarter_spring.csv", index=False)
    # quarter_summer.to_csv("./data/seasonal/quarter_summer.csv", index=False)
    # quarter_autumn.to_csv("./data/seasonal/quarter_autumn.csv", index=False)
    # quarter_winter.to_csv("./data/seasonal/quarter_winter.csv", index=False)
    df = quarter_spring
    # quarter = quarter_4
    # quarter = quarter.drop(columns=["month"])


    # # TODO：进行随机森林rfr
    # rfr_process(quarter)

    # TODO: 对amenity进行拆分并进行分析
    # amenity_analysis(df)

