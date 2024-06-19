import sys,os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainingPipeline:
    def start_train_pip(self):
        try:
            data_ingestion=DataIngestion()
            raw_data_dir=data_ingestion.initiate_data_ingestion()
            return raw_data_dir
        except Exception as e: raise CustomException(e,sys)
    
    def start_transformation_pip(self,raw_data_path):
        try:
            data_transform=DataTransformation(raw_data_path=raw_data_path)
            train_arr,test_arr=data_transform.initiate_data_transform()
            return train_arr,test_arr
        except Exception as e: raise CustomException(e,sys)
    
    def start_model_train_pip(self, train_arr,test_arr):
        try:
            model_train=ModelTrainer()
            model_score=model_train.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
            return model_score
        except Exception as e: raise CustomException(e,sys)
    
    def run_pip(self):
        try:
            raw_data_dir=self.start_train_pip()
            train_arr,test_arr=self.start_transformation_pip(raw_data_path=raw_data_dir)
            accuracy=self.start_model_train_pip(train_arr=train_arr,test_arr=test_arr)
            print(f"training Completed accuracy of best model: {accuracy}")
        except Exception as e: raise CustomException(e,sys)
    
