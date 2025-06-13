import pandas as pd
import numpy as np

def base_features(df):
    # Simple numeric features
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    if "INCOME" in numerics:
        numerics.remove("INCOME")
    if "ID" in numerics:
        numerics.remove("ID")
    return df[numerics].copy()

def preprocess(df):
    feats = base_features(df)
    # Add more feature engineering here if desired
    return feats.fillna(feats.mean())
