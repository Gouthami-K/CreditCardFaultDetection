import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.CreditCardFaultDetection.logger import logging
from src.CreditCardFaultDetection.exception import customexception

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            # Get accuracy scores for train and test data
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Precision
            Precision = precision_score(y_test, y_test_pred)

            # Recall score for test data
            
            recall_score_value = recall_score(y_test, y_test_pred)

            # f1 score for test data
            f1 = f1_score(y_test, y_test_pred)

            # Auc Roc
            AUCROC=roc_auc_score(y_test, y_test_pred)

            # Store scores in the report dictionary
            report[model_name] = {'train_accuracy': train_model_score, 'test_accuracy': test_model_score,'precision_score':Precision, 'recall_score':recall_score_value,'f1_score':f1,'roc_auc_score':AUCROC}

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise customexception(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)

def rename_columns(df):
    try:
        column_mappings = {
            'PAY_0': 'PAY_SEPT',
            'PAY_2': 'PAY_AUG',
            'PAY_3': 'PAY_JUL',
            'PAY_4': 'PAY_JUN',
            'PAY_5': 'PAY_MAY',
            'PAY_6': 'PAY_APR',
            'BILL_AMT1': 'BILL_AMT_SEPT',
            'BILL_AMT2': 'BILL_AMT_AUG',
            'BILL_AMT3': 'BILL_AMT_JUL',
            'BILL_AMT4': 'BILL_AMT_JUN',
            'BILL_AMT5': 'BILL_AMT_MAY',
            'BILL_AMT6': 'BILL_AMT_APR',
            'PAY_AMT1': 'PAY_AMT_SEPT',
            'PAY_AMT2': 'PAY_AMT_AUG',
            'PAY_AMT3': 'PAY_AMT_JUL',
            'PAY_AMT4': 'PAY_AMT_JUN',
            'PAY_AMT5': 'PAY_AMT_MAY',
            'PAY_AMT6': 'PAY_AMT_APR'
        }

        for old_col, new_col in column_mappings.items():
            try:
                df.rename(columns={old_col: new_col}, inplace=True)
            except KeyError:
                # Handle the case where the column doesn't exist in the DataFrame
                logging.info(f"Column '{old_col}' not found in DataFrame.")

    except Exception as e:
        raise customexception(e, sys)
    
def modify_columns(df):
    try:
        # Modify 'EDUCATION' column
        fil_education = (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0)
        df.loc[fil_education, 'EDUCATION'] = 4

        # Modify 'MARRIAGE' column
        fil_marriage = df['MARRIAGE'] == 0
        df.loc[fil_marriage, 'MARRIAGE'] = 3

        return df
    
    except Exception as e:
        raise customexception(e, sys)

# Import necessary libraries
from imblearn.over_sampling import SMOTE

def smote_balance(data):
    try:
        target_column_name = 'default.payment.next.month'
        sm = SMOTE(sampling_strategy='minority', random_state=42)

        logging.info('Dataset shape prior resampling: {}'.format(data.shape[0]))
        X_resampled, y_resampled = sm.fit_resample(X=data.drop(columns=target_column_name), y=data[target_column_name])
        data_resampled = pd.concat([pd.DataFrame(X_resampled, columns=data.drop(columns=target_column_name).columns),
                                    pd.DataFrame(y_resampled, columns=[target_column_name])], axis=1)
        logging.info('Dataset shape after resampling: {}'.format(data_resampled.shape[0]))
        
        return data_resampled

    except Exception as e:
        raise customexception(e, sys)

    
def replace_categories(df):
    try:
        # Check and filter 'SEX' column
        valid_sex_values = {1, 2}
        invalid_sex_values = set(df['SEX'].unique()) - valid_sex_values

        if invalid_sex_values:
            raise ValueError(f"Invalid values found in 'SEX' column: {invalid_sex_values}. Expected values are 1 and 2.")

        df.replace({'SEX': {1: 'MALE', 2: 'FEMALE'},
                    'EDUCATION': {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'},
                    'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}}, inplace=True)
    except Exception as e:
        logging.error(f"Error in replacing categories: {e}")
        raise customexception(e, sys)



def rename_and_drop(df):
    try:
        df['IsDefaulter'] = df['default.payment.next.month']
        df.drop('default.payment.next.month', axis=1, inplace=True)
    except Exception as e:
        logging.error(f"Error in renaming and dropping columns: {e}")
        raise customexception(e, sys)