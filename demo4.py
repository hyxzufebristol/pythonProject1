#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/23 21:14



# # 序列相关性分析
# import numpy as np
# from scipy.stats import pearsonr, spearmanr
#
# data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# data2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# r, p = pearsonr(data1, data2)
# print('Pearson correlation coefficient:', r)
# print('p-value:', p)


# ---------------------------- 3. 序列分析 ---------------------
# 以host做profit_per_month 和sentiment的相关性（相关性）
# 因此要提升sentiment的方法

# 序列分析任务：序列分析一定是某一个变量按照年月组成一个一维的list。Nick的月sentiment的变化情况


# Select the columns to analyze
column1 = 'column1'
column2 = 'column2'

# Calculate the Pearson correlation coefficient
correlation = df[column1].corr(df[column2])