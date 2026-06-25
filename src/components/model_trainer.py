import os 
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

@dataclass
class modeltrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class modeltrainer:
    def __init__(self):
        self.model_trainer_config=modeltrainerconfig()
    def inititate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting training and test data")
            xtrain,ytrain,xtest,ytest=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
            'linear regression':LinearRegression(),
            'support vecotr':SVR(),
            'Ridge':Ridge(),
            'Lasso':Lasso(),
            'knn':KNeighborsRegressor(),
            'decision tree':DecisionTreeRegressor(),
            'random forest':RandomForestRegressor(),
            'xgboost':XGBRegressor(),
            'catboosting':CatBoostRegressor(),
            'adaboost':AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model]
            if best_model_score <0.6:
                raise CustomException("not best model found")
            logging.info("model training done and selecting best model ")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            ypred=best_model.predict(xtest)
            r2=r2_score(ytest,ypred)
            return r2

        except Exception as e:
            raise CustomException(e,sys)


