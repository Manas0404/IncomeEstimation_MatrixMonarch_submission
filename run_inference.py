import pandas as pd
import joblib
from features.feat_engineering import preprocess

# Load the test data
df_test = pd.read_csv('data/Hackathon_bureau_data_50000.csv')

# Preprocess (same as training)
X_test = preprocess(df_test)

# Load your trained model
model = joblib.load('models/my_final_model.pkl')

# Predict
test_preds = model.predict(X_test)

# Attach predictions to 'id' column for submission
submission = pd.DataFrame({'id': df_test['id'], 'target_income': test_preds})

# Save as required (CSV)
submission.to_csv('output/output_sample.csv', index=False)

print("Predictions saved to output/output_sample.csv")
