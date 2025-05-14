import os
import pickle
import numpy as np
import logging
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def create_models():
    """
    Creates five different machine learning models for crop prediction:
    - KNN (K-Nearest Neighbors)
    - Random Forest
    - SVM (Support Vector Machine)
    - Logistic Regression
    - Decision Tree
    
    Returns:
    - Dictionary of initialized models
    """
    models = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'logistic_regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    return models

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Advanced prediction function that uses multiple models to recommend crops
    based on soil and environmental parameters.
    
    Uses a voting ensemble of 5 models:
    - KNN
    - Random Forest
    - SVM
    - Logistic Regression
    - Decision Tree
    
    The final prediction is based on the most frequent prediction among all models.
    If models can't be loaded, falls back to a rule-based system.
    
    Parameters:
    - n: Nitrogen content in soil
    - p: Phosphorus content in soil
    - k: Potassium content in soil
    - temperature: Temperature in Celsius
    - humidity: Humidity percentage
    - ph: pH value of soil
    - rainfall: Rainfall in mm
    
    Returns:
    - String containing the recommended crop
    """
    try:
        # Define directories to search for model files
        model_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'pkl_files'),  # models/pkl_files
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),  # models directory
            os.path.dirname(os.path.dirname(__file__)),  # root directory
        ]
        
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        # Define all possible model file names
        model_file_patterns = [
            # Original naming pattern
            {'knn': 'knn_model.pkl', 
             'random_forest': 'random_forest_model.pkl', 
             'svm': 'svm_model.pkl', 
             'logistic_regression': 'logistic_regression_model.pkl', 
             'decision_tree': 'decision_tree_model.pkl'},
            
            # Alternative naming pattern (from train.py)
            {'knn': 'knn_scratch_model.pkl', 
             'random_forest': 'rf_scratch_model.pkl', 
             'svm': 'svm_scratch_model.pkl', 
             'logistic_regression': 'logistic_scratch_model.pkl', 
             'decision_tree': 'dt_scratch_model.pkl'},
            
            # Generic numbered pattern
            {'knn': 'model_1.pkl', 
             'random_forest': 'model_2.pkl', 
             'svm': 'model_3.pkl', 
             'logistic_regression': 'model_4.pkl', 
             'decision_tree': 'model_5.pkl'}
        ]
        
        # Search for models using all patterns in all directories
        models_found = False
        model_paths = {}
        
        for directory in model_dirs:
            if models_found:
                break
                
            for pattern in model_file_patterns:
                # Check if all model files for this pattern exist in this directory
                if all(os.path.exists(os.path.join(directory, filename)) for filename in pattern.values()):
                    # Found all models in this directory with this pattern
                    for model_name, filename in pattern.items():
                        model_paths[model_name] = os.path.join(directory, filename)
                    models_found = True
                    logging.info(f"Found all model files in {directory} using pattern {pattern}")
                    break
        
        if models_found:
            # Load all models and make predictions
            predictions = []
            model_predictions = {}
            
            for model_name, model_path in model_paths.items():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Handle both raw model and dictionary formats
                    if isinstance(model_data, dict) and 'model' in model_data:
                        # Format: {'model': model_instance, 'scaler': scaler, ...}
                        model = model_data['model']
                        
                        # If there's a scaler, use it
                        if 'scaler' in model_data:
                            input_scaled = model_data['scaler'].transform(input_data)
                        else:
                            input_scaled = input_data
                            
                        # Make prediction
                        if hasattr(model, 'predict_classes'):
                            pred = model.predict_classes(input_scaled)[0]
                        elif hasattr(model, 'predict_batch'):
                            pred = model.predict_batch(input_scaled)[0]
                        else:
                            pred = model.predict(input_scaled)[0]
                            
                        # Decode prediction if needed
                        if 'label_encoder' in model_data and hasattr(model_data['label_encoder'], 'inverse_transform'):
                            pred = model_data['label_encoder'].inverse_transform([pred])[0]
                            
                    else:
                        # Format: direct model instance
                        model = model_data
                        pred = model.predict(input_data)[0]
                    
                    predictions.append(pred)
                    model_predictions[model_name] = pred
                    logging.info(f"{model_name.upper()} prediction: {pred}")
                    
                except Exception as e:
                    logging.error(f"Error with model {model_name}: {str(e)}")
                    
            if predictions:
                # Get the most common prediction (voting)
                prediction_counts = Counter(predictions)
                most_common_prediction = prediction_counts.most_common(1)[0][0]
                
                logging.info(f"Final ensemble prediction: {most_common_prediction}")
                # Return both the final prediction and individual model predictions
                return most_common_prediction, model_predictions
        
        # Try to load single pre-trained model as fallback
        for directory in model_dirs:
            model_path = os.path.join(directory, 'crop_recommendation_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Handle both raw model and dictionary formats
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    
                    # If there's a scaler, use it
                    if 'scaler' in model_data:
                        input_scaled = model_data['scaler'].transform(input_data)
                    else:
                        input_scaled = input_data
                        
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    
                    # Decode prediction if needed
                    if 'label_encoder' in model_data and hasattr(model_data['label_encoder'], 'inverse_transform'):
                        prediction = model_data['label_encoder'].inverse_transform([prediction])[0]
                else:
                    model = model_data
                    prediction = model.predict(input_data)[0]
                
                logging.info(f"Using single model prediction: {prediction}")
                # Return prediction and a simple model_predictions dictionary with just this model
                model_name = os.path.basename(model_path).split('.')[0]
                return prediction, {model_name: prediction}
        
        # If models not found, use basic rule-based system
        logging.warning("Using fallback prediction rules")
        
        # Create a simple rule-based system with explanations
        rule_predictions = {}
        
        # Rules for different pH ranges
        if ph < 5.5:
            rule_predictions["Low pH Rule"] = "rice" if rainfall > 200 and temperature > 25 else "blueberry"
        elif ph < 6.5:
            rule_predictions["Mid-Low pH Rule"] = "maize" if k > 40 and rainfall < 100 else "wheat"
        elif ph < 7.5:
            rule_predictions["Mid-High pH Rule"] = "cotton" if n > 40 and p > 40 else "sunflower"
        else:  # pH >= 7.5
            rule_predictions["High pH Rule"] = "sugarcane" if rainfall > 200 else "chickpea"
            
        # Additional rules based on other parameters
        if temperature > 25 and humidity > 80:
            rule_predictions["Hot & Humid Rule"] = "rice"
        if n > 80 and p > 50 and k > 40:
            rule_predictions["High NPK Rule"] = "banana"
        if rainfall < 100 and temperature > 20:
            rule_predictions["Dry & Warm Rule"] = "chickpea"
            
        # Get the most common prediction from our rules
        rule_values = list(rule_predictions.values())
        final_prediction = max(set(rule_values), key=rule_values.count)
        
        # Return both the prediction and the reasoning
        return final_prediction, rule_predictions
                
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        # If all else fails, return a general recommendation and an error indicator
        return "Unable to make prediction. Please check your input values.", {"Error": "Model loading failed"}
