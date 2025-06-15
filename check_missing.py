import pandas as pd

# Load your data
df = pd.read_csv("data/Hackathon_bureau_data_400.csv")  # Change the path if needed

# Count missing values for all columns
missing_summary = df.isna().sum()

# Show only columns with missing values
print("Columns with missing values:\n", missing_summary[missing_summary > 0])
