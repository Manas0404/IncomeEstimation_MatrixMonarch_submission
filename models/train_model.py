import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from features.feat_engineering import preprocess_fill_minus1, preprocess_fill_mean

def run_and_score(X, y, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse, model

def train():
    df = pd.read_csv("data/Hackathon_bureau_data_50000.csv")
    df = df.dropna(subset=['target_income'])
    y = df['target_income']

    # --- Pick your real continuous columns below ---
    continuous_cols = ['monthly_income', 'credit_score']  # <<<<<<<< CHANGE as per your dataset!

    # Approach 1: Fill all missing with -1
    X1 = preprocess_fill_minus1(df)
    rmse1, model1 = run_and_score(X1, y)
    print(f"[All -1] Validation RMSE: {rmse1:.4f}")

    # Approach 2: Fill only continuous with mean, others -1
    X2 = preprocess_fill_mean(df, continuous_cols)
    rmse2, model2 = run_and_score(X2, y)
    print(f"[Continuous mean, others -1] Validation RMSE: {rmse2:.4f}")

    if rmse1 < rmse2:
        print("Model with all -1 imputation is better.")
        best_model = model1
    else:
        print("Model with mean for continuous features is better.")
        best_model = model2

    import joblib
    joblib.dump(best_model, './models/my_final_model.pkl')
    print("Best model saved at ./models/my_final_model.pkl")
=======
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from features.feat_engineering import preprocess

def train():
    # Load data
    df = pd.read_csv("data/Hackathon_bureau_data_400.csv")
    print("Loaded data shape:", df.shape)

    # Target variable
    if 'target_income' not in df.columns:
        raise Exception("'target_income' column not found! Check your data.")

    # Drop rows with missing target
    df = df.dropna(subset=['target_income'])

    # Preprocess features
    X = preprocess(df)
    y = df['target_income']

    # --- Debug Section: Check for problematic columns ---
    print("Dtypes after preprocess:\n", X.dtypes)
    print("Sample feature values:\n", X.head(10))

    # Check if any object columns slipped through
    obj_cols = X.select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        print("WARNING: Object columns found in features! They will be dropped:", list(obj_cols))
        X = X.drop(columns=obj_cols)

    # Ensure all NaNs are handled (for safety)
    X = X.fillna(X.mean(numeric_only=True))

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Train a basic Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Validation
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse:.4f}")

    # Save model
    import joblib
    joblib.dump(model, './models/my_final_model.pkl')
    print("Model saved at ./models/my_final_model.pkl")

if __name__ == "__main__":
    train()
