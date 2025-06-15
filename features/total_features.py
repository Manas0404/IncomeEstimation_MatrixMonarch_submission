import pandas as pd
from feat_engineering import preprocess

# Load your dataset (adjust path as needed)
df = pd.read_csv('data/Hackathon_bureau_data_400.csv')
#apply for preprocessing
# Apply your preprocessing function
X = preprocess(df)

# Show all feature columns and the total count
print("Feature columns used in the model:")
print(list(X.columns))
print("\nTotal number of features:", X.shape[1])
