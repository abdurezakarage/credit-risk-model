import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column='TransactionStartTime'):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.datetime_column] = pd.to_datetime(X_[self.datetime_column])
        X_['TransactionHour'] = X_[self.datetime_column].dt.hour
        X_['TransactionDay'] = X_[self.datetime_column].dt.day
        X_['TransactionMonth'] = X_[self.datetime_column].dt.month
        X_['TransactionYear'] = X_[self.datetime_column].dt.year
        return X_

class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby('CustomerId').agg(
            TotalTransactionAmount=('Value', 'sum'),
            AverageTransactionAmount=('Value', 'mean'),
            TransactionCount=('TransactionId', 'count'),
            StdDevTransactionAmount=('Value', 'std')
        ).reset_index()
        return agg_df

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.categorical_features:
            X_[col] = self.encoders[col].transform(X_[col].astype(str))
        return X_

class WOETransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.woe = WOE()

    def fit(self, X, y):
        self.woe.fit(X, y)
        return self

    def transform(self, X):
        return self.woe.transform(X)

class FeatureEngineeringPipeline:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.pipeline = None

    def build_pipeline(self):
        # Handle missing values for numeric columns
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_features:
            numeric_features.remove(self.target_col)

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_features = self.df.select_dtypes(include='object').columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        self.pipeline = Pipeline([
            ('time_features', TimeFeatureExtractor()),
            ('preprocess', preprocessor),
        ])
        return self.pipeline

    def transform(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        pipeline = self.build_pipeline()
        X_transformed = pipeline.fit_transform(X, y)
        return X_transformed
