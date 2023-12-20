
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.CreditCardFaultDetection.exception import customexception
from src.CreditCardFaultDetection.logger import logging

from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.CreditCardFaultDetection.utils.utils import save_object,smote_balance,rename_columns,replace_categories, modify_columns

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be scaled
            numerical_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                              'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            
            # Define which columns should be onehot-encoded and which should be scaled
            #categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']
            #numerical_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY',
                              #'BILL_AMT_APR','PAY_AMT_SEPT','PAY_AMT_AUG','PAY_AMT_JUL','PAY_AMT_JUN','PAY_AMT_MAY','PAY_AMT_APR']
            
            # Define the custom ranking for each ordinal variable
            #SEX_categories = ['FEMALE', 'MALE']
            #EDUCATION_categories = ['graduate school','university','high school','others']
            #MARRIAGE_categories = ['married', 'single', 'others']
            #PAY_categories = ['PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                #('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                #('imputer',SimpleImputer(strategy='most_frequent')),
                #('ordinalencoder',OrdinalEncoder(categories=[SEX_categories])),
                #('onehotencoder', OneHotEncoder(sparse_output=False,handle_unknown='ignore')),
                ('scaler',StandardScaler())
                ]
            )
        
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            #preprocessor=ColumnTransformer([
            #('num_pipeline',num_pipeline,numerical_cols),
            #('cat_pipeline',cat_pipeline,categorical_cols)
            #])
            
            return preprocessor
            

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data from CSV files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'default.payment.next.month'
            drop_columns = [target_column_name, 'ID']

            logging.info("Replacing categories")
            # Replace categories
            #replace_categories(train_df)
            #replace_categories(test_df)

            logging.info("Modifying columns")
            # Modify columns
            modify_columns(train_df)
            modify_columns(test_df)

            logging.info("Renaming columns")
            # Rename columns
            #rename_columns(train_df)
            #rename_columns(test_df)

            #logging.info("Rename and drop columns")
            # Rename and drop
            #rename_and_drop(train_df)
            #rename_and_drop(test_df)

            logging.info("Applying SMOTE to balance classes")
            # Apply SMOTE to balance classes
            train_df = train_df.drop(columns=['_id'], axis=1)
            balanced_train_df=smote_balance(train_df)

            test_df = test_df.drop(columns=['_id'], axis=1)
            balanced_test_df=smote_balance(test_df)

            logging.info("Extracting features and target columns")
            # Extract features and target columns
            input_feature_train_df = balanced_train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = balanced_train_df[target_column_name]

            input_feature_test_df = balanced_test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = balanced_test_df[target_column_name]

            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{input_feature_test_df.head().to_string()}')


            logging.info("Applying preprocessing object on training and testing datasets")
            # Apply preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Combining features and target columns into arrays")
            # Combine features and target columns into arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing pickle file")
            # Save preprocessing pickle file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            logging.info("Preprocessing pickle file saved")

            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Exception occurred in the initialize_data_transformation: {e}")
            raise customexception(e, sys)
