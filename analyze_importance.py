import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
rf = joblib.load('models/my_final_model.pkl')

# Load and preprocess data to get feature names
df = pd.read_csv('data/Hackathon_bureau_data_400.csv')
from features.feat_engineering import preprocess

X = preprocess(df)  # Only one value returned
feature_names = X.columns

# --- Random Forest Importance ---
rf_importances = rf.feature_importances_
feat_imp_rf = pd.DataFrame({'feature': feature_names, 'importance': rf_importances})
feat_imp_rf = feat_imp_rf.sort_values(by='importance', ascending=False)
print("\nTop 20 Random Forest Features:\n", feat_imp_rf.head(20))

# --- Plot: Top 20 Random Forest Features ---
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_rf['feature'][:20][::-1], feat_imp_rf['importance'][:20][::-1])
plt.xlabel("Importance")
plt.title("Top 20 Random Forest Feature Importances")
plt.tight_layout()
plt.show()
