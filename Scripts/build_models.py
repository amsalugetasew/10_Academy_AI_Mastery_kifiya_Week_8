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
import pickle  # For saving classical ML models

class ModelTrainer1:
    def __init__(self, df_credit, df_fraud):
        self.df_credit = df_credit
        self.df_fraud = df_fraud
        self.scaler = StandardScaler()  # Standardize features

    def train_test_split1(self):
        df = self.df_credit  # Select the appropriate dataset
        df = self.df_fraud  # Alternatively, use df_fraud if that's your dataset
        target_column = 'class'

        """Prepares data by splitting features and target, scaling features, and reshaping for deep learning"""
        if "signup_time" in df.columns and "purchase_time" in df.columns:
            # Convert datetime columns to Unix timestamps (numeric format)
            df["signup_time"] = pd.to_datetime(df["signup_time"]).astype('int64') // 10**9
            df["purchase_time"] = pd.to_datetime(df["purchase_time"]).astype('int64') // 10**9

            # Feature Engineering: Time difference between signup and purchase
            df["signup_to_purchase_seconds"] = df["purchase_time"] - df["signup_time"]

            # Drop original datetime columns if not needed
            df = df.drop(columns=["signup_time", "purchase_time"])

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column].values  # Convert y to a NumPy array

        # Ensure `y` is reshaped for TensorFlow models
        y = y.reshape(-1, 1)  # Now works since y is a NumPy array

        # Scale features using StandardScaler
        X_scaled = self.scaler.fit_transform(X)

        # Perform train-test split while keeping the DataFrame format
        a, b, c, d = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Convert numpy arrays back to DataFrame for both X and y
        a = pd.DataFrame(a, columns=X.columns)
        b = pd.DataFrame(b, columns=X.columns)
        c = pd.DataFrame(c, columns=[target_column])
        d = pd.DataFrame(d, columns=[target_column])

        return a, b, c, d, df.drop(columns=[target_column])

    def prepare_data(self, df, target_column):
        """Prepares data by splitting features and target, scaling features, and reshaping for deep learning"""
        if "signup_time" in df.columns and "purchase_time" in df.columns:
            # Convert datetime columns to Unix timestamps (numeric format)
            df["signup_time"] = pd.to_datetime(df["signup_time"]).astype('int64') // 10**9
            df["purchase_time"] = pd.to_datetime(df["purchase_time"]).astype('int64') // 10**9

            # Feature Engineering: Time difference between signup and purchase
            df["signup_to_purchase_seconds"] = df["purchase_time"] - df["signup_time"]

            # Drop original datetime columns if not needed
            df = df.drop(columns=["signup_time", "purchase_time"])
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Ensure `y` is correctly reshaped for TensorFlow models
        y = y.reshape(-1, 1)

        # Scale features
        X = self.scaler.fit_transform(X)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_cnn_model(self, input_shape):
        """Convolutional Neural Network (CNN) Model"""
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
        """Recurrent Neural Network (RNN) Model"""
        model = keras.Sequential([
            layers.SimpleRNN(64, activation='relu', input_shape=input_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm_model(self, input_shape):
        """Long Short-Term Memory (LSTM) Model"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.LSTM(32),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate_dl(self, X_train, X_test, y_train, y_test, model, model_name):
        """Train and Evaluate Deep Learning Models"""
        print(f"\nTraining {model_name} model...")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")  
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")  

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Log experiment with MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.tensorflow.log_model(model, model_name)

        # Save deep learning model for deployment
        model.save(f"{model_name}.h5")
        print(f"Model {model_name} saved as {model_name}.h5")
   
   
    def run_experiments(self):
        """Run all experiments for classical ML and Deep Learning models"""
        mlflow.set_experiment("fraud_detection_experiment")

        for dataset_name, df, target in [("Credit Card", self.df_credit, "Class"),("Fraud Data", self.df_fraud, "class")]:
            print(f"\nTraining models for {dataset_name} dataset...")

            # Prepare tabular data for ML models
            X_train, X_test, y_train, y_test = self.prepare_data(df, target)

            # Reshape data for deep learning models
            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

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
                    model.fit(X_train, y_train.ravel())  # `.ravel()` to flatten labels
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"{model_name} Accuracy: {accuracy:.4f}")
                    print(classification_report(y_test, y_pred))

                    # Log experiment with MLflow
                    input_example = X_test[:5]  # A few test samples
                    

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.sklearn.log_model(model, model_name, input_example=input_example)
                    # mlflow.sklearn.log_model(model, model_name)

                    # Save classical ML models
                    with open(f"{dataset_name}_{model_name}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    print(f"Model {dataset_name}_{model_name} saved as {model_name}.pkl")

            # Train Deep Learning Models
            deep_models = {
                "CNN": self.build_cnn_model(input_shape=(X_train_reshaped.shape[1], 1)),
                "RNN": self.build_rnn_model(input_shape=(X_train_reshaped.shape[1], 1)),
                "LSTM": self.build_lstm_model(input_shape=(X_train_reshaped.shape[1], 1))
            }

            for model_name, model in deep_models.items():
                with mlflow.start_run(run_name=f"{dataset_name} - {model_name}"):
                    self.train_and_evaluate_dl(X_train_reshaped, X_test_reshaped, y_train, y_test, model, model_name)
