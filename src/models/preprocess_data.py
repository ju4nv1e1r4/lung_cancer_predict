"""
This script performs data preprocessing for the lung cancer dataset.
It includes steps like renaming columns, encoding categorical variables, 
generating age brackets, and exporting a preprocessed dataset.

Functions are modularized, follow PEP8 standards, and include docstrings.
"""

import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a specified CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse the file at {file_path}.")


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and rename specific columns.

    Args:
        df (pd.DataFrame): Original dataframe.

    Returns:
        pd.DataFrame: Dataframe with renamed columns.
    """
    df.columns = df.columns.str.lower()
    column_mapping = {
        'chronic disease': 'chronic_disease',
        'alcohol consuming': 'alcohol_consuming',
        'shortness of breath': 'shortness_of_breath',
        'swallowing difficulty': 'swallowing_difficulty',
        'chest pain': 'chest_pain'
    }
    df.rename(columns=column_mapping, inplace=True)
    return df


def encode_binary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode binary columns into 0 and 1.

    Args:
        df (pd.DataFrame): Dataframe containing the columns.
        columns (list): List of columns to encode.

    Returns:
        pd.DataFrame: Dataframe with binary columns encoded.
    """
    for label in columns:
        df[label] = (df[label] == 2).astype(int)
    return df


def age_bracket(age: int) -> int:
    """
    Categorize age into predefined brackets.

    Args:
        age (int): Age of the individual.

    Returns:
        int: Bracket number (1-5).
    """
    if age >= 78:
        return 5
    elif age >= 59:
        return 4
    elif age >= 43:
        return 3
    elif age >= 27:
        return 2
    else:
        return 1


def age_bracket_str(bracket: int) -> str:
    """
    Convert age bracket into a generation name.

    Args:
        bracket (int): Bracket number.

    Returns:
        str: Generation name.
    """
    mapping = {
        1: 'generation_Z',
        2: 'millennials',
        3: 'generation_X',
        4: 'baby_boomers',
        5: 'silent_generation'
    }
    return mapping.get(bracket, 'unknown')


def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Perform one-hot encoding for categorical columns.

    Args:
        df (pd.DataFrame): Dataframe with categorical columns.
        columns (list): List of categorical columns to encode.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    ohe = OneHotEncoder(handle_unknown='ignore')
    transformed = ohe.fit_transform(df[columns])
    encoded_df = pd.DataFrame(transformed.toarray(), columns=ohe.get_feature_names_out())
    encoded_df.reset_index(drop=True, inplace=True)
    return encoded_df


def save_data(df: pd.DataFrame, output_path: str):
    """
    Save the dataframe to a CSV file.

    Args:
        df (pd.DataFrame): Dataframe to save.
        output_path (str): Path to save the CSV file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    except IOError as e:
        raise IOError(f"Failed to save data to {output_path}: {e}")


def main():
    """
    Main function to execute the preprocessing steps.
    """
    input_path = 'data/external/survey_lung_cancer.csv'
    output_path = 'data/processed/preprocessed.csv'

    df = load_data(input_path)
    df = preprocess_columns(df)
    df = encode_binary_columns(
        df,
        ['smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
         'fatigue ', 'allergy ', 'wheezing', 'alcohol_consuming', 'coughing',
         'shortness_of_breath', 'swallowing_difficulty', 'chest_pain']
    )
    df['generation'] = df['age'].apply(age_bracket)
    df['gen_flag'] = df['generation'].apply(age_bracket_str)
    df['lung_cancer'] = df['lung_cancer'].replace({'YES': 1, 'NO': 0})

    df_encoded = encode_categorical(df, ['gender', 'gen_flag'])
    df_final = pd.concat([df.reset_index(drop=True), df_encoded], axis=1)

    save_data(df_final, output_path)


if __name__ == '__main__':
    main()
