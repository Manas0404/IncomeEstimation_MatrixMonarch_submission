import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_fill_minus1(df, ignore_cols=None):
    """
    Label-encode all categoricals. Fill ALL missing values with -1.
    """
    df = df.copy()
    if ignore_cols is None:
        ignore_cols = ["id", "INCOME", "target_income"]
    for col in df.columns:
        if col in ignore_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col])
            except Exception:
                df[col] = -1
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore')
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(-1)
    X = X.astype(np.float32)
    return X

def preprocess_fill_mean(df, ignore_cols=None, continuous_cols=None):
    """
    Label-encode all categoricals. For continuous_cols, fill missing with mean; rest with -1.
    """
    df = df.copy()
    if ignore_cols is None:
        ignore_cols = ["id", "INCOME", "target_income"]
    if continuous_cols is None:
        continuous_cols = []
    for col in df.columns:
        if col in ignore_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col])
            except Exception:
                df[col] = -1
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore')
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in X.columns:
        if col in continuous_cols:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(-1)
    X = X.astype(np.float32)
    return X

def preprocess_fill_median(df, ignore_cols=None, continuous_cols=None):
    """
    Label-encode all categoricals. For continuous_cols, fill missing with median; rest with -1.
    """
    df = df.copy()
    if ignore_cols is None:
        ignore_cols = ["id", "INCOME", "target_income"]
    if continuous_cols is None:
        continuous_cols = []
    for col in df.columns:
        if col in ignore_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col])
            except Exception:
                df[col] = -1
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore')
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in X.columns:
        if col in continuous_cols:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(-1)
    X = X.astype(np.float32)
    return X
