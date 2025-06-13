import pandas as pd
import joblib
from features.feat_engineering import preprocess

def predict(test_csv_path, output_csv_path):
    df = pd.read_csv(test_csv_path)
    X = preprocess(df)
    model = joblib.load('./models/my_final_model.pkl')
    preds = model.predict(X)

    # Output CSV: ID, INCOME
    submission = pd.DataFrame({
        "ID": df["ID"] if "ID" in df.columns else range(len(df)),
        "INCOME": preds
    })
    submission.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == '__main__':
    import sys
    predict(sys.argv[1], sys.argv[2])
 