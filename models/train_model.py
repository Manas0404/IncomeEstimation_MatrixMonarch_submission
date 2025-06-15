import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features.feat_engineering import (
    preprocess_fill_minus1,
    preprocess_fill_mean,
    preprocess_fill_median
)

def calc_metrics(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix}RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return rmse, mae, r2

def run_and_score(X, y, random_state=42, label="Model"):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=random_state)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    # XGBoost
    xgb = XGBRegressor(n_estimators=150, max_depth=10, random_state=random_state, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_val)
    # LightGBM
    lgbm = LGBMRegressor(n_estimators=150, max_depth=10, random_state=random_state, n_jobs=-1)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_val)
    # Ensemble
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
        "rf": {"model": rf, "rmse": rmse_rf, "mae": mae_rf, "r2": r2_rf},
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

    # --- Auto-detect continuous columns ---
    num_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    continuous_cols = [c for c in num_cols if df[c].nunique() > 10]
    print("\nAuto-detected continuous columns:", continuous_cols)

    # ---- Run all three imputation strategies ----
    X1 = preprocess_fill_minus1(df, ignore_cols)
    metrics1 = run_and_score(X1, y, label="All -1")
    X2 = preprocess_fill_mean(df, ignore_cols, continuous_cols)
    metrics2 = run_and_score(X2, y, label="Continuous mean, others -1")
    X3 = preprocess_fill_median(df, ignore_cols, continuous_cols)
    metrics3 = run_and_score(X3, y, label="Continuous median, others -1")

    print("\nSummary Table:")
    print("            |   RMSE      |   MAE      |    R²   ")
    print(f"[Ens -1]    | {metrics1['ens']['rmse']:.2f} | {metrics1['ens']['mae']:.2f} | {metrics1['ens']['r2']:.4f}")
    print(f"[Ens mean]  | {metrics2['ens']['rmse']:.2f} | {metrics2['ens']['mae']:.2f} | {metrics2['ens']['r2']:.4f}")
    print(f"[Ens median]| {metrics3['ens']['rmse']:.2f} | {metrics3['ens']['mae']:.2f} | {metrics3['ens']['r2']:.4f}")

    # --- Choose best ensemble (lowest RMSE) ---
    best_idx = np.argmin([
        metrics1['ens']['rmse'],
        metrics2['ens']['rmse'],
        metrics3['ens']['rmse']
    ])
    strategies = [("minus1", metrics1), ("mean", metrics2), ("median", metrics3)]
    strategy_name, best_metrics = strategies[best_idx]
    print(f"\nBest strategy: {strategy_name}")

    best_rf = best_metrics['rf']['model']
    best_xgb = best_metrics['xgb']['model']
    best_lgbm = best_metrics['lgbm']['model']

    # Save models + info for inference
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'rf': best_rf,
        'xgb': best_xgb,
        'lgbm': best_lgbm,
        'fill_strategy': strategy_name,
        'continuous_cols': continuous_cols
    }, './models/my_final_model.pkl')
    print("Best models (RF + XGB + LGBM) saved at ./models/my_final_model.pkl")

if __name__ == "__main__":
    train()
