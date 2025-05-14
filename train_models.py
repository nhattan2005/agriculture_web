"""
Script to train and save all five machine learning models for crop recommendation
This is a command-line utility to train models with the provided dataset
"""

import os
import sys
import logging
from models.train_models import train_and_save_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to train models from CSV data"""
    # Default path for the dataset
    default_data_path = os.path.join(os.path.dirname(__file__), 'data/crop_recommendation.csv')
    
    # Get dataset path from command line argument or use default
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path
    
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at {data_path}")
        print(f"Error: Data file not found at {data_path}")
        print("Please provide a valid path to the training data CSV file.")
        return 1
    
    # Train and save models
    print(f"Using dataset: {data_path}")
    accuracy_scores = train_and_save_models(data_path)
    
    # Print summary
    if accuracy_scores:
        print("\nModel Accuracy Summary:")
        for name, accuracy in accuracy_scores.items():
            print(f"{name.upper()}: {accuracy:.4f}")
        print("\nAll models have been trained and saved successfully.")
        print("You can now use the models for crop prediction.")
        return 0
    else:
        print("Error training models. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())