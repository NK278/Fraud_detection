import sys,os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,classification_report
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')
    expt_accuracy=0.99
    model_config_path=os.path.join('config','model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.utils=MainUtils()
        self.models={
            'GaussianNaiveBayes':GaussianNB(),
            "RandomForest":RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(objective='binary:logistic')
        }
    
    def evaluate_model(self,X,y,models):
        try:
            X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42)
            report={}
            for i in range(len(list(models))):
                model=list(models.values())[i]
                model.fit(X_train,y_train)
                y_train_pred=model.predict(X_train)
                y_test_pred=model.predict(X_test)
                train_accuracy_score=accuracy_score(y_train,y_train_pred)
                test_accuracy_score=accuracy_score(y_test,y_test_pred)
                report[list(models.keys())[i]]=test_accuracy_score
            return report
        except Exception as e : raise CustomException(e,sys)
    
    def get_best_model(self,X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array):
        try:
            model_report:dict=self.evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=self.models)
            print(model_report)
            best_model_accuracy=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_accuracy)]
            best_model_obj=self.models[best_model_name]
            return best_model_name,best_model_accuracy,best_model_obj
        except Exception as e : raise CustomException(e,sys)
    
    def hyperparameter_tuning(self,best_model_obj:object,best_model_name,X_train,y_train)->object:
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid=GridSearchCV(best_model_obj, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=3)
            grid.fit(X_train,y_train)
            best_pr=grid.best_params_
            print("best params are:", best_pr)
            tuned_model=best_model_obj.set_params(**best_pr)
            return tuned_model

        except Exception as e : raise CustomException(e,sys)
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Starting model training on train data')
            X_train, X_test, y_train, y_test=(train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1])
            logging.info('Extracting model_config_path')
            model_report:dict=self.evaluate_model(X=X_train,y=y_train,models=self.models)
            best_model_accuracy=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_accuracy)]
            best_model_obj=self.models[best_model_name]
            best_model=self.hyperparameter_tuning(best_model_obj=best_model_obj,best_model_name=best_model_name,X_train=X_train,y_train=y_train)
            best_model.fit(X_train,y_train)
            y_pred=best_model.predict(X_test)
            best_model_score=accuracy_score(y_test,y_pred)
            print(f'best model name:{best_model_name}, best model accuracy:{best_model_score}')
            if best_model_score<0.5: raise Exception('No Model found for threshold=0.5')
            logging.info('best model found on both datasets')
            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model,
            )
            return best_model_score

        except Exception as e : raise CustomException(e,sys)