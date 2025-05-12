import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle

from model import SVM, MultiClassLogisticRegression, KNN, DecisionTreeClassifierFromScratch, RandomForestClassifierFromScratch

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')
label_encoder = LabelEncoder()
df['crop'] = label_encoder.fit_transform(df['label'])

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['crop']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# List of models to train
models = {
    'svm_scratch_model.pkl': SVM(kernel='polynomial', C=1.0, gamma=0.1, degree=3),
    'knn_scratch_model.pkl': KNN(k=3),  # Ensure k is specified
    'dt_scratch_model.pkl': DecisionTreeClassifierFromScratch(is_continuous_list=[True, True, True, True, True, True, True]),
    'rf_scratch_model.pkl': RandomForestClassifierFromScratch(n_trees=20, max_features=3, is_continuous_list=[True, True, True, True, True, True, True]),  # Use n_trees instead of n_estimators
    'logistic_scratch_model.pkl': MultiClassLogisticRegression()
}

for filename, model in models.items():
    print(f"Training {filename}...")
    
    model.fit(X_train, y_train)

    if isinstance(model, KNN):
        y_pred = model.predict_batch(X_test)
    elif isinstance(model, MultiClassLogisticRegression):
        y_pred = model.predict_classes(X_test)
    else:
        y_pred = model.predict(X_test)

    # Ensure y_pred is a NumPy array
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f"{filename} - Accuracy: {acc:.4f}")

    # Save model
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder
        }, f)

    print(f"Saved: {filename}")