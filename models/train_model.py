import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from features.feat_engineering import preprocess
from math import sqrt

def train():
    df = pd.read_csv('./data/Hackathon_bureau_data_400.csv')
    X = preprocess(df)
    y = df['target_income']


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse:.4f}")

    # Save model
    joblib.dump(model, './models/my_final_model.pkl')
    print("Model saved at ./models/my_final_model.pkl")

if __name__ == '__main__':
    train()
