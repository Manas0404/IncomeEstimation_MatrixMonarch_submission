import os
import pandas as pd
import joblib

# ==== STEP 1: Ensure output directory exists ====
os.makedirs('output', exist_ok=True)

# ==== STEP 2: Load the test data ====
df_test = pd.read_csv('data/Hackathon_bureau_data_50000.csv')

# ==== STEP 3: Import your preprocess function ====
# Make sure you use the SAME one as training!
from features.feat_engineering import preprocess_fill_minus1  # or preprocess_fill_mean, or just preprocess

# Choose the right preprocess function (example uses preprocess_fill_minus1)
X_test = preprocess_fill_minus1(df_test)

# ==== STEP 4: Load the trained model ====
model = joblib.load('models/my_final_model.pkl')

# ==== STEP 5: Predict ====
test_preds = model.predict(X_test)

# ==== STEP 6: Prepare the submission dataframe ====
submission = pd.DataFrame({'id': df_test['id'], 'target_income': test_preds})

# ==== STEP 7: Save predictions ====
submission.to_csv('output/output_sample.csv', index=False)
print("Predictions saved to output/output_sample.csv")

# ==== (Optional) Preview the output ====
print(submission.head(10))
