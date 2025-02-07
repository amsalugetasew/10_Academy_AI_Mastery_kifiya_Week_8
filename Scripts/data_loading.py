import pandas as pd
class CSVReader:
    def __init__(self, file_path):
        """
        Initializes the CSVReader with a file path.

        :param file_path: str, path to the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def read_csv(self):
        """
        Reads the CSV file and stores the DataFrame in the 'data' attribute.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("CSV file successfully loaded.")
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_head(self, n=5):
        """
        Returns the first n rows of the DataFrame.

        :param n: int, number of rows to return (default is 5).
        :return: DataFrame
        """
        if self.data is not None:
            return self.data.head(n)
        else:
            print("Data not loaded. Please load the data using read_csv().")
            return None
    def get_shape(self):
        """
        Returns the shape of the DataFrame.

        :return: tuple (rows, columns)
        """
        if self.data is not None:
            return self.data.shape
        else:
            print("Data not loaded. Please load the data using read_csv().")
            return None

    def get_columns(self):
        """
        Returns the list of column names in the DataFrame.

        :return: list of column names
        """
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            print("Data not loaded. Please load the data using read_csv().")
            return None

