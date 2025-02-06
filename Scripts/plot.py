from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class Visualization:
    def __init__(self, df, df1):
        self.df = df
        self.df1 = df1
    def eda(self):
        corr_matrix = self.df.corr()
        numeric_df = self.df1.select_dtypes(include=['number'])
        corr_matrix1 = numeric_df.corr()
        mask = np.triu(np.ones_like(corr_matrix,dtype=bool), k =1)
        mask1 = np.triu(np.ones_like(corr_matrix1,dtype=bool), k =1)
        plt.figure(figsize=(16,12))
        sns.heatmap(corr_matrix, mask = mask, annot=True, vmax = 1, fmt = ".2f", cmap="RdYlGn")
        plt.title("Correlation Matrix")
        plt.show()
        
        plt.figure(figsize=(16,12))
        sns.heatmap(corr_matrix1, mask = mask1, annot=True, vmax = 1, fmt = ".2f", cmap="RdYlGn")
        plt.title("Correlation Matrix")
        plt.show()
    def histogram(self, df_cleaned):
        # Histograms for all numeric columns
        df_cleaned.hist(figsize=(15, 10), bins=30)
        plt.show()
    def boxplot(self, df):
        plt.figure(figsize=(15, 5))
        sns.boxplot(data=df.iloc[:, 1:-2])  # Exclude 'Time' and 'Class'
        plt.xticks(rotation=90)
        plt.show()
    def target_variable_analysis(self, df):
        sns.countplot(x=df['Class'])
        plt.title("Class Distribution (Fraud vs Non-Fraud)")
        plt.show()
    def target_variable_merge(self, df):
        sns.countplot(x=df['class'])
        plt.title("Class Distribution (Fraud vs Non-Fraud)")
        plt.show()
    def pair_plot_analysis(self, df):
        sns.pairplot(df[['V1', 'V2', 'V3', 'V4', 'Class']], hue='Class')
        plt.show()
    def Boxplots_for_Amount_vs_Class(self, df):
        sns.boxplot(x='Class', y='Amount', data=df)
        plt.show()
    def Outlier_Detection(self, df):
        z_scores = df.iloc[:, 1:-2].apply(zscore)
        outliers = (z_scores > 3).sum().sort_values(ascending=False)
        print(outliers)  # Shows count of outliers per column
    def Fraud_Rates_by_Categorical_Features(self, df):
        for col in ["source", "browser", "sex"]:
            fraud_rates = df.groupby(col)['class'].mean().sort_values()

            plt.figure(figsize=(8, 6))
            colors = sns.color_palette("husl", len(fraud_rates))  # Generates distinct colors
            bars = fraud_rates.plot(kind="bar", color=colors)

            plt.title(f"Fraud Rate by {col}")
            plt.xlabel(col)  # Remove x-axis label
            plt.ylabel("Fraud Rate")
            plt.xticks(rotation=45)

            # **Add Data Values on Bars**
            for bar in bars.patches:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                    bar.get_height(),  # Y position (bar height)
                    f"{bar.get_height():.2f}",  # Format value with 2 decimals
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
                )

            plt.show()
    
    def Fraud_Rates_by_Categorical_Features1(self, df):
        for col in ["source", "browser", "sex"]:
            fraud_rates = df.groupby(col)['class'].mean().sort_values()

            plt.figure(figsize=(8, 4))
            colors = plt.cm.viridis(np.linspace(0, 1, len(fraud_rates)))  # Gradient colors
            
            fraud_rates.plot(kind="bar", color=colors)
            plt.title(f"Fraud Rate by {col}")
            plt.xlabel(col)
            plt.ylabel("Fraud Rate")
            plt.xticks(rotation=45)
            plt.show()




