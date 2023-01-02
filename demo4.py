#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/23 21:14
import pandas as pd

from config import reduce_mem_usage

"""
which requires python 3.7 to intall the esential packages.
"""

# # 序列分析预测
# # predict Paul这个host的下个月profit
# df_concat = pd.read_csv('./data/sentiment/df_positive.csv')
# df_concat = reduce_mem_usage(df_concat)
# df = df_concat[df_concat['host_name']=='Paul']
# df.to_csv("./data/prediction/Paul_dataset.csv",index=False)
# monthly_profit_of_Paul = df.groupby(['year','month'])['profit_per_month'].mean()
# monthly_profit_of_Paul.to_csv("./data/prediction/monthly_profit_of_Paul.csv",index=False)

df = pd.read_csv("./data/prediction/Paul_profit.csv")

