import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataPreprocessor:
    def __init__(self, df, df1, df2):
        self.df = df
        self.df1 = df1
        self.df2 = df2

    def handle_missing_values(self):
        self.df.dropna(inplace=True)
        self.df1.dropna(inplace=True)
        self.df2.dropna(inplace=True)
        print("Missing values handled.")

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        self.df1.drop_duplicates(inplace=True)
        self.df2.drop_duplicates(inplace=True)
        print("Duplicates removed.")

    def correct_data_types(self):
        self.df1["signup_time"] = pd.to_datetime(self.df1["signup_time"])
        self.df1["purchase_time"] = pd.to_datetime(self.df1["purchase_time"])
        self.df1["ip_address"] = self.df1["ip_address"].astype(float)
        self.df2["lower_bound_ip_address"] = self.df2["lower_bound_ip_address"].astype(float)
        self.df2["upper_bound_ip_address"] = self.df2["upper_bound_ip_address"].astype(float)
        print("Data types corrected.")

    def eda(self):
        plt.figure(figsize=(12,6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.show()
        
    def merge_datasets(self):
        self.df1 = self.df1.merge(self.df2, how="left", 
                                  left_on="ip_address", 
                                  right_on="lower_bound_ip_address")
        self.df1.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"], inplace=True)
        print("Datasets merged.")
    
    def feature_engineering(self):
        self.df1["transaction_count"] = self.df1.groupby("user_id")["user_id"].transform("count")
        self.df1["hour_of_day"] = self.df1["purchase_time"].dt.hour
        self.df1["day_of_week"] = self.df1["purchase_time"].dt.dayofweek
        print("Feature engineering completed.")
    
    def normalize_and_scale(self):
        scaler = MinMaxScaler()
        self.df["Amount"] = scaler.fit_transform(self.df[["Amount"]])
        print("Normalization applied.")
    
    def encode_categorical_features(self):
        self.df1 = pd.get_dummies(self.df1, columns=["source", "browser", "sex"], drop_first=True)
        print("Categorical features encoded.")
    
    def preprocess(self):
        self.handle_missing_values()
        self.remove_duplicates()
        self.correct_data_types()
        self.eda()
        self.merge_datasets()
        self.feature_engineering()
        self.normalize_and_scale()
        self.encode_categorical_features()
        print("Data preprocessing complete.")
    
    def get_processed_data(self):
        return self.df, self.df1, self.df2

# Example usage:
# preprocessor = DataPreprocessor(df, df1, df2)
# preprocessor.preprocess()
# df_clean, df1_clean, df2_clean = preprocessor.get_processed_data()
