# -*- coding: utf-8 -*- 
# @Time : 2022/10/28 19:08 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
import pandas as pd
import pydotplus as pydotplus
from IPython.core.display import Image
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from six import StringIO
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import export_graphviz
from tqdm import tqdm
warnings.filterwarnings('ignore')
from tools import *

pd.set_option('display.max_columns', None)

# ------------------------- data clean --------------------
# data = pd.read_csv("eureka_data_final_2019-01-01_2019-03-01.csv")
# data = reduce_mem_usage(data)
# data.dropna(axis=0,inplace = True)
# data.to_csv('data.csv',index=0)
# ------------------------- data process-------------------
data = pd.read_csv('data.csv', nrows=10000)

numerical_fea = list(data.select_dtypes(exclude=['object','category']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data.columns)))
category_fea.remove('date')

model_oe = OrdinalEncoder()
string_data_array = model_oe.fit_transform(data[category_fea])
string_data_pd = pd.DataFrame(string_data_array,columns=category_fea)
data[category_fea]=string_data_pd


features = [f for f in data.columns if f not in ['converted_in_7days','date']]
x_train = data[features]
y_train = data["converted_in_7days"].astype('int')

ros = RandomOverSampler()
x_train, y_train = ros.fit_resample(x_train,y_train)

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
# dot_data = StringIO()
# export_graphviz(pipe.named_steps['classifier'].estimators_[0],
#                 out_file=dot_data)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('tree.png')
# Image(graph.create_png())

# ----------------------- feature importance -----------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = x_train.columns

for f in range(x_train.shape[1]):
    print("{}) {} is {}".format(f + 1, feat_labels[indices[f]], importances[indices[f]]))

# # show the pic
# plt.figure()
# plt.title("feature importance")
# plt.bar(range(10), importances[indices][:10])
# plt.xticks(range(10), feat_labels[:10], rotation=90)
# plt.gcf().subplots_adjust(bottom=0.6)
# plt.show()


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
#
# best_estimator = cls.best_estimator_
# print(best_estimator)
# print(cls.best_score_)

# ---------------------- model performance ----------------
# print('Training RMSE:{}'.format(np.sqrt(mean_squared_error(train_y, train_pred))))
# print('Test RMSE:{}'.format(np.sqrt(mean_squared_error(val_y, val_pred))))
# print('Training R-squared:{}'.format(r2_score(train_y, train_pred)))
# print('Test R-squared:{}'.format(r2_score(val_y, val_pred)))