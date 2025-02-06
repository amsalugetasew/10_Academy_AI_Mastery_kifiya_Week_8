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