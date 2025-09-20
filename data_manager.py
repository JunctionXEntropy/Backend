import pandas as pd

# Define a constant for the data file path
DATA_PATH = 'oulu_hospital_patients.csv'

def load_patient_data():
    """
    Loads the patient dataset from the specified CSV file.

    Handles FileNotFoundError if the file does not exist.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        print(f"Error: The data file was not found at '{DATA_PATH}'. Please ensure the file exists.")
        return None