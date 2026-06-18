import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

print("Loading data...")
df = pd.read_csv('insurance.csv')

# Encoding exactly as in app.py
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

print("Training Random Forest Regressor on scikit-learn 1.7.2...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

with open('rf_tuned.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model retrained and saved successfully as rf_tuned.pkl!")
