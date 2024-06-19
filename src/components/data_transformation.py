import sys,os
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['trans_date_trans_time'] = X['trans_date_trans_time'].astype(str)
        X['trans_date_trans_time'] = X['trans_date_trans_time'].str.strip('-')
        X['tran_date'] = X['trans_date_trans_time'].str[:10]
        X['tran_time'] = X['trans_date_trans_time'].str[10:]
        X.drop(columns=['trans_date_trans_time'], inplace=True)
        X['tran_yr'] = (X['tran_date'].str[:4]).astype(int)
        X['tran_m'] = (X['tran_date'].str[5:7]).astype(int)
        X['tran_day'] = (X['tran_date'].str[8:]).astype(int)
        X['tran_hr'] = (X['tran_time'].str[2]).astype(int)
        X['tran_min'] = (X['tran_time'].str[4:5]).astype(int)
        X['tran_sec'] = (X['tran_time'].str[7:]).astype(int)
        X.drop(columns=['tran_time', 'tran_date'], inplace=True)
        return X
        
class GenderMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gen_map={'M': 0, 'F': 1}):
        self.gen_map = gen_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['gender'] = X['gender'].map(self.gen_map)
        return X

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=['merchant', 'first', 'last', 'street', 'city','job']):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.drop(columns=self.columns_to_drop, inplace=True)
        return X

class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='state', prefix='state'):
        self.column = column
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        one_hot_encoded = pd.get_dummies(X[self.column], prefix=self.prefix)
        X = pd.concat([X, one_hot_encoded], axis=1)
        X.drop(columns=[self.column], inplace=True)
        return X

class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='category'):
        self.column = column
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X[self.column])
        return self

    def transform(self, X):
        X[f'{self.column}_label_encoded'] = self.label_encoder.transform(X[self.column]).astype(int)
        X.drop(columns=[self.column], inplace=True)
        return X

class DOBConversionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['dob'] = pd.to_datetime(X['dob'])
        X['int_dob'] = X['dob'].astype('int64') // 10**9
        X.drop(columns=['dob'], inplace=True)
        return X

class TransactionHashingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def hash_transaction(transaction):
            hashed = hashlib.sha256(transaction.encode()).hexdigest()
            return hashed
        
        X['hashed_trans_num'] = X['trans_num'].apply(hash_transaction)
        vectorizer = CountVectorizer(analyzer='char')
        X_transformed = vectorizer.fit_transform(X['trans_num'])
        df_tokenized = pd.DataFrame(X_transformed.toarray(), columns=vectorizer.get_feature_names_out())
        X = pd.concat([X, df_tokenized], axis=1)
        X.drop(columns=['trans_num', 'hashed_trans_num'], inplace=True)
        return X
    
class BoolToIntConversionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bool_columns = X.select_dtypes(include=bool)
        bool_columns_int = bool_columns.astype(int)
        X.drop(columns=bool_columns.columns, inplace=True)
        X = pd.concat([X, bool_columns_int], axis=1)
        return X

@dataclass
class DataTransformconfig:
    transformed_train_file_path = os.path.join('artifacts', 'train.npy')
    transformed_test_file_path = os.path.join('artifacts', 'test.npy')
    transformed_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self,raw_data_path):
        self.raw_data_path=raw_data_path
        self.data_transform_config=DataTransformconfig()
        self.utils=MainUtils()

    def  get_data_transformer_object(self):
        try:
            date_transform_step=('date_transform',DateTransformer())
            gen_mapping_step=('gen_map',GenderMappingTransformer())
            drop_col_step=('drop',DropColumnsTransformer())
            ohe=('OHE',OneHotEncodingTransformer())
            label_step=('label_en',LabelEncodingTransformer())
            dob_dtep=('dob_int',DOBConversionTransformer())
            hashed_step=('hashed',TransactionHashingTransformer())
            bool_int_step=('bool_int',BoolToIntConversionTransformer())
            preprosesor=Pipeline(
                steps=[
                    date_transform_step,
                    gen_mapping_step,
                    drop_col_step,
                    ohe,
                    label_step,
                    dob_dtep,
                    hashed_step,
                    bool_int_step
                ]
            )
            return preprosesor
        except Exception as e : raise CustomException(e,sys)
    
    def initiate_data_transform(self):
        def transform_df(X):
            def hash_transaction(transaction):
                hashed = hashlib.sha256(transaction.encode()).hexdigest()
                return hashed
            X['trans_date_trans_time'] = X['trans_date_trans_time'].astype(str)
            X['trans_date_trans_time'] = X['trans_date_trans_time'].str.strip('-')
            X['tran_date'] = X['trans_date_trans_time'].str[:10]
            X['tran_time'] = X['trans_date_trans_time'].str[10:]
            X.drop(columns=['trans_date_trans_time'], inplace=True)
            X['tran_yr'] = (X['tran_date'].str[:4]).astype(int)
            X['tran_m'] = (X['tran_date'].str[5:7]).astype(int)
            X['tran_day'] = (X['tran_date'].str[8:]).astype(int)
            X['tran_hr'] = (X['tran_time'].str[2]).astype(int)
            X['tran_min'] = (X['tran_time'].str[4:5]).astype(int)
            X['tran_sec'] = (X['tran_time'].str[7:]).astype(int)
            X.drop(columns=['tran_time', 'tran_date'], inplace=True)
            columns_to_drop=['merchant', 'first', 'last', 'street', 'city','job']
            X.drop(columns=columns_to_drop, inplace=True)
            X['gender'] = X['gender'].map({'M': 0, 'F': 1})
            one_hot_encoded = pd.get_dummies(X['state'], prefix='state')
            X = pd.concat([X, one_hot_encoded], axis=1)
            X.drop(columns=['state'], inplace=True)
            label_encoder = LabelEncoder()
            label_encoder.fit(X['category'])
            col='category'
            X[f'{col}_label_encoded'] = label_encoder.transform(X[col]).astype(int)
            X.drop(columns=[col], inplace=True)
            X['dob'] = pd.to_datetime(X['dob'])
            X['int_dob'] = X['dob'].astype('int64') // 10**9
            X.drop(columns=['dob'], inplace=True)
            X['hashed_trans_num'] = X['trans_num'].apply(hash_transaction)
            vectorizer = CountVectorizer(analyzer='char')
            X_transformed = vectorizer.fit_transform(X['trans_num'])
            df_tokenized = pd.DataFrame(X_transformed.toarray(), columns=vectorizer.get_feature_names_out())
            X = pd.concat([X, df_tokenized], axis=1)
            X.drop(columns=['trans_num', 'hashed_trans_num'], inplace=True)
            bool_columns = X.select_dtypes(include=bool)
            bool_columns_int = bool_columns.astype(int)
            X.drop(columns=bool_columns.columns, inplace=True)
            X = pd.concat([X, bool_columns_int], axis=1)
            return X
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")
        try:
            dataframe=pd.read_csv(self.raw_data_path)
            dataframe=transform_df(dataframe)
            logging.info(f'len of dataset: {len(dataframe)}')
            X=dataframe.drop(labels=['is_fraud'],axis=1)
            logging.info(f'len of X:{len(X)}')
            y=dataframe['is_fraud']
            logging.info(f'len of y:{len(y)}')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            logging.info(f'len of x_train,x_test: {len(X_train)},{len(X_test)}')
            logging.info(f'len of y_train,y_test: {len(y_train)},{len(y_test)}')
            # prep=self.get_data_transformer_object()
            X_train_scaled=StandardScaler().fit_transform(X_train)
            X_test_Scaled=StandardScaler().fit_transform(X_test)
            logging.info(f'len of scaled datasets: {len(X_train_scaled)}, {len(X_test_Scaled)}')
            logging.info(f'len of y_train,y_test: {len(y_train)},{len(y_test)}')
            preprocessor_path = self.data_transform_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path,
                                   obj=transform_df)
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_Scaled, np.array(y_test)]

            return train_arr,test_arr

        except Exception as e : raise CustomException(e,sys)
        