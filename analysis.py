import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Synthesize Travel Choice Dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'trip_distance_miles': np.random.uniform(0.5, 30, n_samples),
    'household_income': np.random.choice([1, 2, 3, 4], n_samples),
    'transit_availability': np.random.choice([0, 1], n_samples),
    'urban_density_score': np.random.uniform(1, 10, n_samples),
})

# Target Mode: 0 = Walk, 1 = Public Transit, 2 = Personal Vehicle
conditions = [
    data['trip_distance_miles'] < 2.0,
    (data['trip_distance_miles'] >= 2.0) & (data['transit_availability'] == 1) & (data['urban_density_score'] > 5)
]
choices = [0, 1]
data['mode_choice'] = np.select(conditions, choices, default=2)

# 2. Train LightGBM Classifier
X = data.drop(columns=['mode_choice'])
y = data['mode_choice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = lgb.LGBMClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# 3. Model Evaluation
y_pred = clf.predict(X_test)
print("--- Travel Mode Choice Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Walk', 'Transit', 'Drive']))
