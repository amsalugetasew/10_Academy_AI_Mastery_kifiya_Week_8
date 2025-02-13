import pandas as pd
import shap
import joblib
import pickle
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

class ModelExplainability:
    
    def __init__(self, model_path, X_train, y_train, X_test, use_joblib=True):
        """
        Initializes the ModelExplainability class and loads a saved model.
        
        :param model_path: Path to the saved model (joblib or pickle format)
        :param X_train: Training data
        :param y_train: Training labels
        :param X_test: Test data
        :param use_joblib: Whether to use joblib for loading (True) or pickle (False)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
        # Load the model
        if use_joblib:
            self.model = joblib.load(model_path)  # Load the model using joblib
        else:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)  # Load the model using pickle
        
        self.explainer_shap = shap.TreeExplainer(self.model)  # SHAP Tree Explainer for tree-based models

    def shap_summary_plot(self):
        """
        Generate SHAP Summary Plot showing the most important features.
        """
        shap_values = self.explainer_shap.shap_values(self.X_train)
        shap.summary_plot(shap_values[1], self.X_train)  # 1 for positive class (fraud)
    
    def shap_force_plot(self, instance_index=0):
        """
        Generate SHAP Force Plot to explain the prediction for a specific instance.
        
        :param instance_index: Index of the instance to explain (default is 0)
        """
        shap_values = self.explainer_shap.shap_values(self.X_train)
        shap.initjs()  # Initialize JS for SHAP visualization
        shap.force_plot(self.explainer_shap.expected_value[1], shap_values[1][instance_index], self.X_train.iloc[instance_index])
    
    def shap_dependence_plot(self, feature_name):
        """
        Generate SHAP Dependence Plot to show feature relationships with the model output.
        
        :param feature_name: The feature to plot (e.g., 'age', 'income')
        """
        shap_values = self.explainer_shap.shap_values(self.X_train)
        shap.dependence_plot(feature_name, shap_values[1], self.X_train)
    
    def lime_explanation(self, instance_index=0, num_features=5):
        """
        Generate LIME explanation for a specific instance.
        
        :param instance_index: Index of the instance to explain (default is 0)
        :param num_features: Number of most important features to display (default is 5)
        """
        # Ensure X_train and y_train are properly formatted
        if isinstance(self.X_train, pd.DataFrame) and isinstance(self.y_train, pd.Series):
            print("X_train is a DataFrame and y_train is a Series")
        else:
            print("X_train and y_train types are not compatible.")

        print(f"Shape of X_train: {self.X_train.shape}")
        print(f"Shape of y_train: {self.y_train.shape}")
        
        # Initialize the LIME explainer
        lime_explainer = LimeTabularExplainer(
            self.X_train.values, mode='classification', training_labels=self.y_train.values
        )
        
        # Access the specific instance from X_test (make sure it's a DataFrame)
        try:
            instance = self.X_test.iloc[instance_index]  # Get the row at the instance_index
            print(f"Instance selected: {instance}")
        except Exception as e:
            print(f"Error selecting instance: {e}")
            return
        
        # Explain the instance (ensure we're passing a 1D array of feature values)
        exp = lime_explainer.explain_instance(instance.values, self.model.predict_proba, num_features=num_features)
        
        # Display the explanation plot
        exp.as_pyplot_figure()
        plt.show()

    def lime_explanation1(self, instance_index=0, num_features=5):
        """
        Generate LIME explanation for a specific instance and display feature importance.
        
        :param instance_index: Index of the instance to explain (default is 0)
        :param num_features: Number of most important features to display (default is 5)
        """
        # Ensure X_train and y_train are properly formatted
        if isinstance(self.X_train, pd.DataFrame) and isinstance(self.y_train, pd.Series):
            print("X_train is a DataFrame and y_train is a Series")
        else:
            print("X_train and y_train types are not compatible.")

        print(f"Shape of X_train: {self.X_train.shape}")
        print(f"Shape of y_train: {self.y_train.shape}")
        
        # Initialize the LIME explainer
        lime_explainer = LimeTabularExplainer(
            self.X_train.values, mode='classification', training_labels=self.y_train.values
        )
        
        # Access the specific instance from X_test (make sure it's a DataFrame)
        try:
            instance = self.X_test.iloc[instance_index]  # Get the row at the instance_index
            print(f"Instance selected: {instance}")
        except Exception as e:
            print(f"Error selecting instance: {e}")
            return
        
        # Explain the instance (ensure we're passing a 1D array of feature values)
        exp = lime_explainer.explain_instance(instance.values, self.model.predict_proba, num_features=num_features)
        
        # Get the feature importances
        feature_importances = exp.as_list()
        
        # Extract feature names and their corresponding importances
        feature_names = [f[0] for f in feature_importances]
        feature_values = [f[1] for f in feature_importances]
        
        # Plot the feature importance as a bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_values, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'LIME Feature Importance for Instance {instance_index}')
        plt.show()


