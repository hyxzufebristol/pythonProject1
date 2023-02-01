#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/23 21:14
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt

from config import reduce_mem_usage

"""
1. pip install pystan
2. conda install -c conda-forge fbprophet
which requires python 3.7 to intall the esential packages.

you could find somne reference here.
https://www.kaggle.com/getting-started/151267
"""

# # 序列分析预测
# # predict Paul这个host的下个月profit
# df_concat = pd.read_csv('./data/sentiment/df_positive.csv')
# df_concat = reduce_mem_usage(df_concat)
# df = df_concat[df_concat['host_name']=='Paul']
# df.to_csv("./data/prediction/Paul_dataset.csv",index=False)
# monthly_profit_of_Paul = df.groupby(['year','month'])['profit_per_month'].mean()
# monthly_profit_of_Paul.to_csv("./data/prediction/monthly_profit_of_Paul.csv",index=False)

df = pd.read_csv("./data/prediction/Paul_profit.csv",encoding='gbk')
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'date': 'ds', 'profit_per_month': 'y'})
# df = df.set_index('ds')

# Fit the Prophet model
m = Prophet(
    changepoint_prior_scale=0.3,
    changepoint_range=0.8
            # ,mcmc_samples=500
            )

m.fit(df)

# Create a dataframe for future predictions
future = m.make_future_dataframe(periods=60)

# Make the forecasts
forecast = m.predict(future)

# Plot the forecast
m.plot(forecast)
plt.show()
m.plot_components(forecast)
plt.show()
