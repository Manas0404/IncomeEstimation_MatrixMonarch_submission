import pandas as pd
import numpy as np
import joblib

def predict(X):
    model_names = joblib.load('./models/model_list.pkl')
    preds = []
    for name in model_names:
        model = joblib.load(f'./models/model_{name}.pkl')
        preds.append(model.predict(X))
    preds = np.array(preds)
    ensemble_preds = preds.mean(axis=0)
    # Also get feature importances (just from RF for now)
    feature_imp = joblib.load('./models/model_rf.pkl').feature_importances_
    return ensemble_preds, feature_imp
