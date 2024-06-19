import sys,os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join('artifacts')
    raw_data_path:str=os.path.join(data_ingestion_dir,'new_fraudTrain.csv')

class DataIngestion:
    def __init__(self) :
        self.data_ing_config=DataIngestionConfig()
        self.utils=MainUtils()
    def export_col_as_df(self,col_name,db):
        try:
            mongo_client=MongoClient("mongodb+srv://nishchalgaur2003:zatfem-wymxyX-kyvqy3@cluster0.kvvgt6j.mongodb.net/?retryWrites=true&w=majority")
            col=mongo_client[db][col_name]
            df=pd.DataFrame(list(col.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=['_id'],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e: raise CustomException(e,sys) from e
    def export_data_into_raw_data_dir(self)->pd.DataFrame:
        try:
            logging.info('Exporting data from mongodb')
            raw_data_dir=self.data_ing_config.data_ingestion_dir
            os.makedirs(raw_data_dir,exist_ok=True)
            raw_data_path=self.data_ing_config.raw_data_path
            logging.info(f'Saving exported data into feature store file path:{raw_data_path}')
            dataset=self.export_col_as_df(db='Fraud_Detection2',col_name='Project_SML1')
            dataset.to_csv(raw_data_path,index=False)
        except Exception as e: raise CustomException(e,sys) from e

    def initiate_data_ingestion(self)->Path:
        logging.info('Entered initiate_data_ingestion method of  DataIngestionConfig class')
        try:
            self.export_data_into_raw_data_dir()
            logging.info('Got data from mongodb')
            logging.info('Exitited initiate_data_ingestion method of  DataIngestionConfig class')
            return self.data_ing_config.raw_data_path
        except Exception as e: raise CustomException(e,sys) from e
