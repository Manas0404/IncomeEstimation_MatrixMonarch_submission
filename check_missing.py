import pandas as pd
import numpy as np

# Your real column order
columns = [
    'id'
] + [f'var_{i}' for i in range(76)] + [
    'var_74', 'var_75',
    'gender', 'marital_status', 'city', 'state', 'residence_ownership', 'target_income',
    'device_model', 'device_category', 'platform', 'device_manufacturer'
]

# 10 dummy rows for demo
dummy_data = []
for i in range(10):
    row = []
    row.append(f"DUMMY_{i+1}")  # id
    row += list(np.random.uniform(-1, 1, 76))  # var_0 to var_75
    row += [
        np.random.choice(['J-High Risk', 'D-Very Low Risk', 'H-Medium Risk', 'E-Low Risk', 'K-High Risk', 'I-Medium Risk', 'B-Very Low Risk', 'C-Very Low Risk']), # var_74
        np.random.choice(['PERFORM CONSUMER 2.0', 'NOT SCORED', 'SUB PRIME']),  # var_75
        np.random.choice(['MALE', 'FEMALE']),  # gender
        np.random.choice(['SINGLE', 'MARRIED', 'DIVORCED']),  # marital_status
        f"City_{np.random.randint(1, 10)}",  # city
        f"State_{np.random.randint(1, 5)}",  # state
        np.random.choice(['OWN', 'RENT', 'FAMILY']),  # residence_ownership
        np.random.randint(5000, 30000),  # target_income
        f"Model_{np.random.randint(1,5)}",  # device_model
        np.random.choice(['SMART PHONE', 'FEATURE PHONE']),  # device_category
        np.random.choice(['ANDROID', 'IOS']),  # platform
        np.random.choice(['samsung', 'xiaomi', 'vivo', 'apple', 'oneplus'])  # device_manufacturer
    ]
    dummy_data.append(row)

# Create DataFrame
df_dummy = pd.DataFrame(dummy_data, columns=columns)
df_dummy.to_csv("data/dummy_test_data.csv", index=False)
