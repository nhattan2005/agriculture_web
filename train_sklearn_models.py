import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_models(csv_file_path, output_dir='models/pkl_files'):
    """
    Train and save 5 different machine learning models for crop prediction.
    
    Parameters:
    - csv_file_path: Path to the CSV training data
    - output_dir: Directory to save the trained models
    
    Returns:
    - Dictionary with model names as keys and accuracy scores as values
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        logging.info(f"Loading data from {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['crop'] = label_encoder.fit_transform(df['label'])
        
        # Features and target
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['crop']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a DataFrame with scaled features to preserve column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42
        )
        
        # Define models to train
        models = {
            'knn_model.pkl': KNeighborsClassifier(n_neighbors=5),
            'random_forest_model.pkl': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm_model.pkl': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
            'logistic_regression_model.pkl': LogisticRegression(max_iter=1000, multi_class='multinomial'),
            'decision_tree_model.pkl': DecisionTreeClassifier(random_state=42)
        }
        
        # Train and save each model
        results = {}
        for filename, model in models.items():
            logging.info(f"Training {filename}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[filename] = accuracy
            logging.info(f"{filename} - Accuracy: {accuracy:.4f}")
            
            # Save model with scaler and label encoder
            model_path = os.path.join(output_dir, filename)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'label_encoder': label_encoder
                }, f)
            
            logging.info(f"Saved: {model_path}")
        
        # Also save a single unified model (the most accurate one)
        best_model_name = max(results, key=results.get)
        best_model_path = os.path.join(output_dir, best_model_name)
        
        # Copy as crop_recommendation_model.pkl
        unified_model_path = os.path.join(output_dir, 'crop_recommendation_model.pkl')
        with open(best_model_path, 'rb') as src, open(unified_model_path, 'wb') as dst:
            dst.write(src.read())
        
        logging.info(f"Created unified model at {unified_model_path} based on {best_model_name}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        return {}

if __name__ == "__main__":
    # Path to CSV file (check in several locations)
    possible_paths = [
        'data/Crop_recommendation.csv',
        'Crop_recommendation.csv',
        'attached_assets/Crop_recommendation.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path:
        logging.info(f"Found CSV file at {csv_path}")
        results = train_and_save_models(csv_path)
        
        if results:
            best_model = max(results, key=results.get)
            logging.info(f"Best model: {best_model} with accuracy {results[best_model]:.4f}")
        else:
            logging.error("No models were trained successfully")
    else:
        logging.error("Crop recommendation CSV file not found. Please provide a valid path.")