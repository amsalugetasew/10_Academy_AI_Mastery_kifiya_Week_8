import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
class ModelTrainer:
    def __init__(self, df_credit, df_fraud):
        self.df_credit = df_credit
        self.df_fraud = df_fraud
    
    def prepare_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model, model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Log experiment with MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)
    
    def run_experiments(self):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "MLP": MLPClassifier()
        }
        
        mlflow.set_experiment("fraud_detection_experiment")
        
        for dataset_name, df, target in [("Credit Card", self.df_credit, "Class"), ("Fraud Data", self.df_fraud, "Class")]:
            print(f"\nTraining models for {dataset_name} dataset...")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target)
            
            for model_name, model in models.items():
                with mlflow.start_run(run_name=f"{dataset_name} - {model_name}"):
                    self.train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name)



