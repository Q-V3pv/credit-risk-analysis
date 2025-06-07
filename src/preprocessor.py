import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    Limpia el DataFrame:
    - Eliminar duplicados
    - Eliminar missing values 
    """
    df = df.drop_duplicates()
 
    df = df.dropna()
    return df

def encode_features(df, categorical_cols):
    """
    Codifica variables categóricas con LabelEncoder.
    """
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def split_features_target(df, target_col):
    """
    Separa características (X) y etiqueta (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
