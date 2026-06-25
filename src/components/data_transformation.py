import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
import os 
@dataclass
class datatransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
class datatranformation:
    def __init__(self):
        self.data_transformation_config=datatransformationconfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns=['math_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                         ('scaler',StandardScaler())])
            
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one hot encoder',OneHotEncoder())

                ]
            )
            logging.info("encoding and scaling completed")

            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),
                                            ('cat_pipeline',cat_pipeline,categorical_columns)])
            logging.info("column tranformer")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading test and train data")
            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column="total score"
            num_columns="math_score"
            input_train_df=train_df.drop(columns=[target_column],axis=1)
            target_train=train_df[target_column]
            input_test_df=test_df.drop(columns=[target_column],axis=1)
            target_test=test_df[target_column]

            scaled_train=preprocessing_obj.fit_transform(input_train_df)
            scaled_test=preprocessing_obj.transform(input_test_df)

            train_arr=np.c_[scaled_train,np.array(target_train)]
            test_arr=np.c_[scaled_test,np.array(target_test)]
            logging.info("saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)


