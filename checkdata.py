import pandas as pd
df = pd.read_csv('data/Hackathon_bureau_data_400.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, df[col].unique()[:20])  # Show first 20 unique values
