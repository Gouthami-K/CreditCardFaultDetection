import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from src.CreditCardFaultDetection.utils.utils import load_object

class ModelEvaluation:
    def __init__(self):
        pass
    
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)  # Accuracy
        precision = precision_score(actual, pred)  # Precision
        recall = recall_score(actual, pred)  # Recall
        f1 = f1_score(actual, pred)  # F1 score
        return accuracy, precision, recall, f1

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/gouthamikrishnamurthy/CreditCardFaultDetection.mlflow")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run():

                predicted_classes = model.predict(X_test)

                (accuracy, precision, recall, f1) = self.eval_metrics(y_test, predicted_classes)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                #This condition is for dagshub
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                #This condition is for local    
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise e
