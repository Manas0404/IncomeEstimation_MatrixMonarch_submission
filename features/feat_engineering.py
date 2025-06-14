import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def data_health_check(df):
    """
    Checks for missing values in columns and returns a warning list.
    """
    warnings = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            warnings.append(f"{col}: {n_missing} missing")
    return warnings

def preprocess(df):
    """
    Final feature engineering:
    - Label-encode ALL categoricals (object columns and non-numeric categoricals)
    - Forces all columns to be numeric float32
    - Fills missing values with -1
    """
    df = df.copy()

    # List of columns to ignore (ID and target)
    ignore = ["id", "INCOME", "target_income"]

    # Label encode all object/categorical columns (including weird ones like var_74, etc.)
    for col in df.columns:
        if col in ignore:
            continue
        # If not already numeric, label encode (as string!)
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col])
            except Exception as e:
                # fallback if error in encoding, replace with -1
                print(f"Label encoding failed for {col}: {e}")
                df[col] = -1

    # Drop ignored columns if still present
    X = df.drop(columns=[c for c in ignore if c in df.columns], errors='ignore')

    # Final force numeric conversion (if anything slipped through)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(-1)
    X = X.astype(np.float32)
    return X
