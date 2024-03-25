import pandas as pd

def retrieve_data(csv_file):
    """
    Retrieve data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the retrieved data.
    """
    return pd.read_csv(csv_file)
