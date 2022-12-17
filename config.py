# -*- coding: utf-8 -*- 
# @Time : 2022/12/17 13:01 
# @Author : YeMeng 
# @File : config.py 
# @contact: 876720687@qq.com
import pandas as pd

data_dir = "/Users/yemeng/hyx/20221116FinalPoj/data/"

# location of folders
data_in = data_dir + "raw/"
data_out = data_dir + "clean/"

# 运行更多行的代码
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)