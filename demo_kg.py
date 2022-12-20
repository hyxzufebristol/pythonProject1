#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/20 20:37
# Statistics
import pandas as pd
import numpy as np
import math as mt

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

px.defaults.width = 1200
px.defaults.height = 800
# plotly.io Settings for both plotly.graph_objects and plotly.express
pio.templates.default = "plotly_white" # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
"""
pio.kaleido.scope.default_format = 'svg'
pio.kaleido.scope.default_scale = 1
"""

# Data Preprocessing - Standardization, Encoding, Imputation
from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.preprocessing import Normalizer # Normalization
from sklearn.preprocessing import OneHotEncoder # One-hot Encoding
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from category_encoders import MEstimateEncoder # Target Encoding
from sklearn.preprocessing import PolynomialFeatures # Create Polynomial Features
from sklearn.impute import SimpleImputer # Imputation

# Exploratory Data Analysis - Feature Engineering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

# Modeling - ML Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Modeling - Algorithms
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ML - Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# ML - Tuning
import optuna
#from sklearn.model_selection import GridSearchCV

# Settings
# Settings for Seaborn
sns.set_theme(context='notebook', style='ticks', palette="bwr_r", font_scale=0.7, rc={"figure.dpi":240, 'savefig.dpi':240})


class ETL_pipeline:
    def __init__(self, data_frame):
        self.df = data_frame

    # Data type transformation
    def _transformation(self, data_frame):
        df = data_frame
        # Convert dollar columns from object to float
        # Remove '$' and ','
        dollar_cols = ['price', 'weekly_price', 'monthly_price', 'extra_people', 'security_deposit', 'cleaning_fee']
        for dollar_col in dollar_cols:
            df[dollar_col] = df[dollar_col].replace('[\$,]', '', regex=True).astype(float)
        # Convert dollar columns from object to float
        # Remove '%'
        percent_cols = ['host_response_rate', 'host_acceptance_rate']
        for percent_col in percent_cols:
            df[percent_col] = df[percent_col].replace('%', '', regex=True).astype(float)

        # Replace the following values in property_type to Unique space due to small sample size
        unique_space = ["Barn",
                        "Boat",
                        "Bus",
                        "Camper/RV",
                        "Treehouse",
                        "Campsite",
                        "Castle",
                        "Cave",
                        "Dome House",
                        "Earth house",
                        "Farm stay",
                        "Holiday park",
                        "Houseboat",
                        "Hut",
                        "Igloo",
                        "Island",
                        "Lighthouse",
                        "Plane",
                        "Ranch",
                        "Religious building",
                        "Shepherd’s hut",
                        "Shipping container",
                        "Tent",
                        "Tiny house",
                        "Tipi",
                        "Tower",
                        "Train",
                        "Windmill",
                        "Yurt",
                        "Riad",
                        "Pension",
                        "Dorm",
                        "Chalet"]
        df.property_type = df.property_type.replace(unique_space, "Unique space", regex=True)

        # Convert 't', 'f' to 1, 0
        tf_cols = ['host_is_superhost', 'instant_bookable', 'require_guest_profile_picture',
                   'require_guest_phone_verification']
        for tf_col in tf_cols:
            df[tf_col] = df[tf_col].replace('f', 0, regex=True)
            df[tf_col] = df[tf_col].replace('t', 1, regex=True)

        return df

    # Parse listings
    def parse_listings(self):
        """Parse listings.
        """
        df = self.df
        df = self._transformation(df)
        return df

    def parse_reviews(self):
        """Parse reviews.
        """
        df = self.df
        df.date = pd.to_datetime(df.date)
        return df

    # Parse calendar
    def parse_calender(self):
        """Paser calendar.
        """
        df = self.df
        # Convert date from object to datetime
        df.date = pd.to_datetime(df.date)
        # Convert price from object to float
        # Convert '$' and ',' to ''
        df.price = df.price.replace('[\$,]', '', regex=True).astype(float)

        # Convert 't', 'f' to 1, 0
        df['available'] = df['available'].replace('f', 0, regex=True)
        df['available'] = df['available'].replace('t', 1, regex=True)

        return df


listings = pd.read_csv("./data/clean/df_clean.csv", delimiter=",")
reviews = pd.read_csv("./data/raw/reviews.csv", delimiter=",")
listings = ETL_pipeline(listings).parse_listings()
reviews = ETL_pipeline(reviews).parse_reviews()



