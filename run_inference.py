import pandas as pd
import numpy as np
import joblib
from features.feat_engineering import preprocess_fill_minus1, preprocess_fill_mean

# Load test data
df_test = pd.read_csv('data/Hackathon_bureau_data_400.csv')
bundle = joblib.load('models/my_final_model.pkl')

fill_strategy = bundle.get('fill_strategy', 'minus1')
continuous_cols = bundle.get('continuous_cols', [])

# Preprocess with the same strategy as the best model
if fill_strategy == 'minus1':
    X_test = preprocess_fill_minus1(df_test, ignore_cols=["id", "INCOME", "target_income"])
else:
    X_test = preprocess_fill_mean(df_test, ignore_cols=["id", "INCOME", "target_income"], continuous_cols=continuous_cols)

# Predict with ensemble
rf_pred = bundle['rf'].predict(X_test)
xgb_pred = bundle['xgb'].predict(X_test)
lgbm_pred = bundle['lgbm'].predict(X_test)
final_pred = (rf_pred + xgb_pred + lgbm_pred) / 3

submission = pd.DataFrame({'id': df_test['id'], 'target_income': final_pred})
submission.to_csv('output/output_sample.csv', index=False)
print("Predictions saved to output/output_sample.csv")

# If test labels are available, 
# print metrics
if 'target_income' in df_test.columns:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = df_test['target_income']
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    mae = mean_absolute_error(y_true, final_pred)
    r2 = r2_score(y_true, final_pred)
    print(f"Test set metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
