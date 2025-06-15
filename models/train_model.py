import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from features.feat_engineering import preprocess_fill_minus1, preprocess_fill_mean

def calc_metrics(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix}RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return rmse, mae, r2

def run_and_score(X, y, random_state=42, label="Model"):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=random_state)

    # --- Hyperparameter tuning for RandomForest (randomized, quick) ---
    rf_param = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf, rf_param, n_iter=4, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=random_state, verbose=0)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_val)

    # --- XGBoost ---
    xgb = XGBRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_val)

    # --- LightGBM ---
    lgbm = LGBMRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_val)

    # --- Ensemble (Average) ---
    ens_pred = (rf_pred + xgb_pred + lgbm_pred) / 3

    print(f"\nResults for {label}:")
    print("  [RandomForest]")
    rmse_rf, mae_rf, r2_rf = calc_metrics(y_val, rf_pred, prefix="    ")
    print("  [XGBoost]")
    rmse_xgb, mae_xgb, r2_xgb = calc_metrics(y_val, xgb_pred, prefix="    ")
    print("  [LightGBM]")
    rmse_lgbm, mae_lgbm, r2_lgbm = calc_metrics(y_val, lgbm_pred, prefix="    ")
    print("  [Ensemble (Avg)]")
    rmse_ens, mae_ens, r2_ens = calc_metrics(y_val, ens_pred, prefix="    ")

    return {
        "rf": {"model": rf_best, "rmse": rmse_rf, "mae": mae_rf, "r2": r2_rf},
        "xgb": {"model": xgb, "rmse": rmse_xgb, "mae": mae_xgb, "r2": r2_xgb},
        "lgbm": {"model": lgbm, "rmse": rmse_lgbm, "mae": mae_lgbm, "r2": r2_lgbm},
        "ens": {"rmse": rmse_ens, "mae": mae_ens, "r2": r2_ens}
    }

def train():
    data_path = "data/Hackathon_bureau_data_50000.csv"
    df = pd.read_csv(data_path)
    print("Loaded data shape:", df.shape)
    if 'target_income' not in df.columns:
        raise Exception("'target_income' column not found! Check your data.")
    df = df.dropna(subset=['target_income'])
    y = df['target_income']

    ignore_cols = ["id", "INCOME", "target_income"]

    # --- AUTO-DETECT CONTINUOUS COLUMNS ---
    num_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    continuous_cols = [c for c in num_cols if df[c].nunique() > 10]
    print("\nAuto-detected continuous columns:", continuous_cols)

    # --- All -1 strategy ---
    X1 = preprocess_fill_minus1(df, ignore_cols)
    metrics1 = run_and_score(X1, y, label="All -1")

    # --- Mean for continuous, -1 for rest strategy --
    X2 = preprocess_fill_mean(df, ignore_cols, continuous_cols)
    metrics2 = run_and_score(X2, y, label="Continuous mean, others -1")

    print("\nSummary Table:")
    print("          |   RMSE      |   MAE      |    R²   ")
    print(f"[RF -1]   | {metrics1['rf']['rmse']:.2f} | {metrics1['rf']['mae']:.2f} | {metrics1['rf']['r2']:.4f}")
    print(f"[XGB -1]  | {metrics1['xgb']['rmse']:.2f} | {metrics1['xgb']['mae']:.2f} | {metrics1['xgb']['r2']:.4f}")
    print(f"[LGBM-1]  | {metrics1['lgbm']['rmse']:.2f} | {metrics1['lgbm']['mae']:.2f} | {metrics1['lgbm']['r2']:.4f}")
    print(f"[Ens -1]  | {metrics1['ens']['rmse']:.2f} | {metrics1['ens']['mae']:.2f} | {metrics1['ens']['r2']:.4f}")
    print(f"[RF mean] | {metrics2['rf']['rmse']:.2f} | {metrics2['rf']['mae']:.2f} | {metrics2['rf']['r2']:.4f}")
    print(f"[XGB mean]| {metrics2['xgb']['rmse']:.2f} | {metrics2['xgb']['mae']:.2f} | {metrics2['xgb']['r2']:.4f}")
    print(f"[LGBMmean]| {metrics2['lgbm']['rmse']:.2f} | {metrics2['lgbm']['mae']:.2f} | {metrics2['lgbm']['r2']:.4f}")
    print(f"[Ens mean]| {metrics2['ens']['rmse']:.2f} | {metrics2['ens']['mae']:.2f} | {metrics2['ens']['r2']:.4f}")

    # Save best model by lowest ensemble RMSE
    if metrics1['ens']['rmse'] < metrics2['ens']['rmse']:
        print("\nEnsemble with all -1 imputation is better.")
        best_rf = metrics1['rf']['model']
        best_xgb = metrics1['xgb']['model']
        best_lgbm = metrics1['lgbm']['model']
        fill_strategy = 'minus1'
    else:
        print("\nEnsemble with mean for continuous features is better.")
        best_rf = metrics2['rf']['model']
        best_xgb = metrics2['xgb']['model']
        best_lgbm = metrics2['lgbm']['model']
        fill_strategy = 'mean'

    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump({'rf': best_rf, 'xgb': best_xgb, 'lgbm': best_lgbm, 'fill_strategy': fill_strategy, 'continuous_cols': continuous_cols}, './models/my_final_model.pkl')
    print("Best models (RF + XGB + LGBM) saved at ./models/my_final_model.pkl")

if __name__ == "__main__":
    train()
