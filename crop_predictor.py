import os
import pickle
import numpy as np
import pandas as pd
import logging
from collections import Counter
from model import KNN, MultiClassLogisticRegression

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Predicts the recommended crop using an ensemble of five custom models.
    
    Parameters:
    - n: Nitrogen content in soil (kg/ha)
    - p: Phosphorus content in soil (kg/ha)
    - k: Potassium content in soil (kg/ha)
    - temperature: Temperature in Celsius
    - humidity: Humidity percentage
    - ph: pH value of soil
    - rainfall: Rainfall in mm
    
    Returns:
    - String containing the recommended crop
    """
    try:
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                                 columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        # Use project root directory for model files
        model_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Log the model directory
        logging.debug(f"Looking for model files in: {model_dir}")
        
        # Define model files and names
        model_files = [
            'svm_scratch_model.pkl',
            'knn_scratch_model.pkl',
            'dt_scratch_model.pkl',
            'rf_scratch_model.pkl',
            'logistic_scratch_model.pkl'
        ]
        model_names = [
            'SVM Scratch',
            'KNN Scratch',
            'Decision Tree Scratch',
            'Random Forest Scratch',
            'Logistic Regression Scratch'
        ]
        
        # Check if all models exist
        models_exist = all(os.path.exists(os.path.join(model_dir, file_name)) for file_name in model_files)
        
        if models_exist:
            # Load models and make predictions
            predictions = []
            for model_name, file_name in zip(model_names, model_files):
                model_path = os.path.join(model_dir, file_name)
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = model_data['model']
                        scaler = model_data['scaler']
                        label_encoder = model_data['label_encoder']
                    
                    # Scale input data
                    input_scaled = scaler.transform(input_data)
                    
                    # Predict (handle different prediction methods)
                    logging.debug(f"Attempting prediction with {model_name}")
                    if isinstance(model, KNN):
                        pred = model.predict(input_scaled[0])
                    elif isinstance(model, MultiClassLogisticRegression):
                        pred = model.predict_classes(input_scaled)[0]
                    else:
                        pred = model.predict(input_scaled)[0]
                    
                    # Decode prediction
                    pred = label_encoder.inverse_transform([pred])[0]
                    predictions.append(pred)
                    logging.info(f"{model_name} prediction: {pred}")
                
                except Exception as e:
                    logging.error(f"Prediction failed for {model_name}: {str(e)}")
                    continue
            
            # Check if any predictions were successful
            if predictions:
                # Get the most common prediction (voting)
                prediction_counts = Counter(predictions)
                most_common_prediction = prediction_counts.most_common(1)[0][0]
                logging.info(f"Final ensemble prediction: {most_common_prediction}")
                return most_common_prediction
            else:
                logging.error("No models produced valid predictions")
                return "Unable to make prediction. All models failed."
        
        else:
            missing_files = [f for f in model_files if not os.path.exists(os.path.join(model_dir, f))]
            logging.error(f"Not all model files found. Missing: {missing_files}")
            return "Unable to make prediction. Model files missing."
                
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return "Unable to make prediction. Please check your input values."