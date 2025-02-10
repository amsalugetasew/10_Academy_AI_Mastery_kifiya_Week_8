
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.tensorflow

class ModelTrainer1:
    def __init__(self, df_credit, df_fraud):
        self.df_credit = df_credit
        self.df_fraud = df_fraud
        self.scaler = StandardScaler()  # Standardize features for deep learning

    def prepare_data(self, df, target_column):
        X = df.drop(columns=[target_column]).values  # Convert DataFrame to NumPy array
        y = df[target_column].values

        # Scale features
        X = self.scaler.fit_transform(X)

        # Reshape for deep learning models
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))  # Add a single channel for CNN/RNN

        return train_test_split(X, y, test_size=0.2, random_state=42), X_reshaped

    def build_cnn_model(self, input_shape):
        """ Convolutional Neural Network (CNN) Model """
        model = keras.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_rnn_model(self, input_shape):
        """ Recurrent Neural Network (RNN) Model """
        model = keras.Sequential([
            layers.SimpleRNN(64, activation='relu', input_shape=input_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm_model(self, input_shape):
        """ Long Short-Term Memory (LSTM) Model """
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.LSTM(32),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate_dl(self, X_train, X_test, y_train, y_test, model, model_name):
        """ Train and Evaluate Deep Learning Models """
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Log experiment with MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.tensorflow.log_model(model, model_name)

    def run_experiments(self):
        """ Run all experiments for classical ML and Deep Learning models """
        mlflow.set_experiment("fraud_detection_experiment")

        for dataset_name, df, target in [("Credit Card", self.df_credit, "Class"), ("Fraud Data", self.df_fraud, "class")]:
            print(f"\nTraining models for {dataset_name} dataset...")

            # Prepare tabular data for ML models and reshaped data for deep learning models
            (X_train, X_test, y_train, y_test), X_train_reshaped = self.prepare_data(df, target)

            # Train Classical ML Models
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neural_network import MLPClassifier

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "MLP": MLPClassifier()
            }

            for model_name, model in models.items():
                with mlflow.start_run(run_name=f"{dataset_name} - {model_name}"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"{model_name} Accuracy: {accuracy:.4f}")
                    print(classification_report(y_test, y_pred))

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.sklearn.log_model(model, model_name)

            # Train Deep Learning Models
            deep_models = {
                "CNN": self.build_cnn_model(input_shape=(X_train_reshaped.shape[1], 1)),
                "RNN": self.build_rnn_model(input_shape=(X_train_reshaped.shape[1], 1)),
                "LSTM": self.build_lstm_model(input_shape=(X_train_reshaped.shape[1], 1))
            }

            for model_name, model in deep_models.items():
                with mlflow.start_run(run_name=f"{dataset_name} - {model_name}"):
                    self.train_and_evaluate_dl(X_train_reshaped, X_train_reshaped, y_train, y_test, model, model_name)

