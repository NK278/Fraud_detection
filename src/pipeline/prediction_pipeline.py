import shutil
import os,sys
import pandas as pd
from src.logger import logging
from src.constant import *
from src.utils.main_utils import MainUtils
from src.exception import CustomException
from flask import request
from dataclasses import dataclass

@dataclass
class PredictionPipelineconfig:
    pred_output_name:str='predictions'
    pred_file_name:str='pred.csv'
    trained_model_path:str=os.path.join('artifacts','model.pkl')
    preprocessor_path:str=os.path.join('artifacts','preprocessor.pkl')
    prep_path:str=os.path.join(pred_output_name,pred_file_name)


class PredPipeline:
    def __init__(self,request:request):
        self.request=request
        self.utils=MainUtils()
        self.pipelineconfig=PredictionPipelineconfig()
    
    def save_input_files(self)->str:
        try:
            pred_file_input_dir='pred_artifacts'
            os.makedirs(pred_file_input_dir,exist_ok=True)
            input_csvfile=self.request.files['file']
            pred_file_path=os.path.join(pred_file_input_dir,input_csvfile.filename)
            input_csvfile.save(pred_file_path)
            return pred_file_path

        except Exception as e: raise CustomException(e,sys)

    def predict(self,features):
        try:
            model_path=self.pipelineconfig.trained_model_path
            preprocessor_path=self.pipelineconfig.preprocessor_path
            model=self.utils.load_object(file_path=model_path)
            prep=self.utils.load_object(file_path=preprocessor_path)
            transformed_features=prep.transform(features)
            preds=model.predict(transformed_features)
            return preds
        except Exception as e: raise CustomException(e,sys)

    def get_pred_df(self,input_dfpath:pd.DataFrame):
        try:
            pred_col_name:str='is_fraud_pred'
            input_df:pd.DataFrame=pd.read_csv(input_dfpath)
            # input_df =  input_df.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_df.columns else input_df
            predictions=self.predict(input_df)
            input_df[pred_col_name]=[p for p in predictions]
            os.makedirs(self.pipelineconfig.pred_output_name,exist_ok=True)
            input_df.to_csv(self.pipelineconfig.prep_path)
        except Exception as e: raise CustomException(e,sys)
    
    def run_pip(self):
        try:
            input_filepath=self.save_input_files()
            self.get_pred_df(input_dfpath=input_filepath)
            return self.pipelineconfig
        except Exception as e:raise CustomException(e,sys)
