import pandas as pd
import numpy as np
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
        return self.df,self.df1,self.df2
        

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        self.df1.drop_duplicates(inplace=True)
        self.df2.drop_duplicates(inplace=True)
        print("Duplicates removed.")
        return self.df,self.df1,self.df2

    def correct_data_types(self):
        self.df1["signup_time"] = pd.to_datetime(self.df1["signup_time"])
        self.df1["purchase_time"] = pd.to_datetime(self.df1["purchase_time"])
        self.df1["ip_address"] = self.df1["ip_address"].astype(float)
        self.df2["lower_bound_ip_address"] = self.df2["lower_bound_ip_address"].astype(float)
        self.df2["upper_bound_ip_address"] = self.df2["upper_bound_ip_address"].astype(float)
        print("Data types corrected.")
        return self.df1,self.df2
        
    def merge_datasets(self):
        # Ensure IP data is sorted for merge_asof()
        self.df2 = self.df2.sort_values("lower_bound_ip_address")

        # Perform the merge using `merge_asof()`
        self.df1 = self.df1.sort_values("ip_address")  # Ensure df1 is sorted before merging
        self.df1 = pd.merge_asof(self.df1, self.df2, left_on="ip_address", right_on="lower_bound_ip_address")

        # Drop unnecessary columns
        self.df1 = self.df1.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])

        # self.df1 = self.df1.merge(self.df2, how="left", 
        #                           left_on="ip_address", 
        #                           right_on="lower_bound_ip_address")
        # self.df1.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"], inplace=True)
        print("Datasets merged.")
        return self.df1
    
    def feature_engineering(self, df):
        df["transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek
        print("Feature engineering completed.")
        return df
    
    def normalize_and_scale(self,df):
        scaler = MinMaxScaler()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
        print("Normalization applied.")
        return df
    
    def encode_categorical_features(self, df1):
        categorical_columns = ["source", "browser", "sex","country"]  # Define categorical columns
    
        for col in categorical_columns:
            df1[col] = pd.Categorical(df1[col]).codes  # Encode each column
        print("Categorical features encoded.")
        return df1
    
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
