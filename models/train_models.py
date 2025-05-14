import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from models.crop_predictor import create_models

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def train_and_save_models(data_path):
    """
    Train and save all five machine learning models for crop prediction
    
    Parameters:
    - data_path: Path to the CSV training data with columns:
      N, P, K, temperature, humidity, ph, rainfall, label
    
    Returns:
    - Dictionary with model names as keys and accuracy scores as values
    """
    try:
        # Load and prepare the data
        logging.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Assuming the CSV has these columns:
        # N, P, K, temperature, humidity, ph, rainfall, label
        X = data.iloc[:, :-1]  # All features except the last column
        y = data.iloc[:, -1]   # Last column as target
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Get all models
        models = create_models()
        
        # Train and save each model
        accuracy_scores = {}
        for name, model in models.items():
            logging.info(f"Training {name.upper()} model...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            accuracy = model.score(X_test, y_test)
            accuracy_scores[name] = accuracy
            logging.info(f"{name.upper()} accuracy: {accuracy:.4f}")
            
            # Save the model
            model_filename = os.path.join(os.path.dirname(__file__), f"{name}_model.pkl")
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Saved {name} model to {model_filename}")
        
        return accuracy_scores
    
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        return {}

if __name__ == "__main__":
    # Example usage - replace with actual data path
    data_path = "CropAdvisor/data/crop_recommendation.csv"
    
    if os.path.exists(data_path):
        accuracy_scores = train_and_save_models(data_path)
        
        # Print summary
        print("\nModel Accuracy Summary:")
        for name, accuracy in accuracy_scores.items():
            print(f"{name.upper()}: {accuracy:.4f}")
    else:
        logging.error(f"Data file not found at {data_path}")
        print(f"Error: Data file not found at {data_path}")
        print("Please provide a valid path to the training data CSV file.")