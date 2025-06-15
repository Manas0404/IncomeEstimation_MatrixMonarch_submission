# Income Estimation - Matrix Monarch Hackathon

## Overview

This project predicts **target income** from diverse demographic, behavioral, and device-related features using an ensemble of advanced machine learning models.  
It was built for the Matrix Monarch AI Hackathon.

- **Models used:** Random Forest, XGBoost, LightGBM (ensemble)
- **Feature engineering:** Robust label encoding, missing value imputation (`-1`, mean, or median)
- **Metrics reported:** RMSE, MAE, R² on validation and test data
- **Submission format:** `output/output_sample.csv` as required

---

## Folder Structure

.
├── data/
│ ├── Hackathon_bureau_data_50000.csv # Main training dataset
│ ├── Hackathon_bureau_data_400.csv # Example/test dataset
├── models/
│ ├── my_final_model.pkl # Saved ensemble model (auto-generated)
├── features/
│ └── feat_engineering.py # All feature engineering utilities
├── output/
│ └── output_sample.csv # Submission predictions (auto-generated)
├── main.py # Entrypoint to train the model
├── run_inference.py # Script to run predictions for submission
├── requirements.txt # Required Python packages
├── README.md # Project instructions (this file)

markdown
Copy
Edit

---

## Requirements

- **Python 3.8+**
- See `requirements.txt`.  
  Core dependencies:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `joblib`

Install with:

```bash
pip install -r requirements.txt
Quickstart
1. Training the Model
To train the ensemble on your main dataset:

bash
Copy
Edit
python main.py train
This will train the ensemble with three imputation strategies (-1, mean, median).

It saves the best model and preprocessing details to models/my_final_model.pkl.

Validation scores (RMSE, MAE, R²) will be printed in the terminal.

2. Running Inference (Generating Submission)
To generate predictions on a test or holdout set (e.g., for the jury):

bash
Copy
Edit
python run_inference.py
This uses models/my_final_model.pkl and the best-matching preprocessing.

Output: Predictions are saved to output/output_sample.csv
Format:

id	target_income
...	...

If the test file contains true target_income, performance metrics are also printed.

File Descriptions
main.py
Main training launcher. Accepts command-line arg: train

models/train_model.py
Contains training logic, ensemble construction, auto-detects best imputation strategy.

features/feat_engineering.py
Feature engineering, robust label encoding, multiple missing value strategies.

run_inference.py
Loads the trained model, preprocesses input, saves predictions for submission.

requirements.txt
List of all required libraries (Python packages).

data/
Place all training and testing CSV files here.

output/
Output predictions will be saved here.

models/
Trained models are saved here (auto-generated).

Customization
Adding/Removing Features:
Adjust column processing in features/feat_engineering.py

Changing Imputation Strategies:
Add more logic to feat_engineering.py or tweak in train_model.py

Hyperparameter Tuning:
Extend search grids in train_model.py for better performance.

Acknowledgements
Competition data and structure provided by Matrix Monarch Hackathon organizers.

Thanks to scikit-learn, XGBoost, and LightGBM communities for open-source tools.

Contact
For any questions, contact:
Manas Ranjan Rout
Email: routmanas49@gmail.com

Best of luck to all participants!

yaml
Copy
Edit

---

**Feel free to customize the author info, add team members, or tweak any details as needed!**  
Let me know if you want a lighter/shorter version or extra instructions (for AWS, etc).
