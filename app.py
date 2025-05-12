import os
import pickle
import numpy as np
import pandas as pd
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from crop_predictor import predict_crop
from model import KNN, MultiClassLogisticRegression

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    model_predictions = None
    
    if request.method == 'POST':
        try:
            # Get form data
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Store input values in session for display
            session['input_data'] = {
                'nitrogen': n,
                'phosphorus': p,
                'potassium': k,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            # Make prediction using your ensemble
            prediction_result = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            
            # Get individual model predictions
            model_dir = os.path.dirname(__file__)  # Use project root
            model_predictions = {}
            model_names = ['SVM Scratch', 'KNN Scratch', 'Decision Tree Scratch', 'Random Forest Scratch', 'Logistic Regression Scratch']
            file_names = ['svm_scratch_model.pkl', 'knn_scratch_model.pkl', 'dt_scratch_model.pkl', 
                          'rf_scratch_model.pkl', 'logistic_scratch_model.pkl']
            
            input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                                     columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            for idx, file_name in enumerate(file_names):
                model_path = os.path.join(model_dir, file_name)
                logging.debug(f"Checking model file: {model_path}")
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                            model_instance = model_data['model']
                            scaler = model_data['scaler']
                            label_encoder = model_data['label_encoder']
                        
                        # Scale input data
                        input_scaled = scaler.transform(input_data)
                        
                        # Predict (handle different prediction methods)
                        logging.debug(f"Attempting individual prediction with {model_names[idx]}")
                        if isinstance(model_instance, KNN):
                            pred = model_instance.predict(input_scaled[0])
                        elif isinstance(model_instance, MultiClassLogisticRegression):
                            pred = model_instance.predict_classes(input_scaled)[0]
                        else:
                            pred = model_instance.predict(input_scaled)[0]
                            # Handle float64 predictions
                            if isinstance(pred, float):
                                pred = int(round(pred))
                        
                        # Decode prediction
                        pred = label_encoder.inverse_transform([pred])[0]
                        model_predictions[model_names[idx]] = pred
                    except Exception as model_err:
                        logging.error(f"Error with model {file_name}: {str(model_err)}")
                else:
                    logging.error(f"Model file not found: {model_path}")
            
            flash(f'Recommended crop: {prediction_result}', 'success')
                
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            flash(f'Error making prediction: {str(e)}', 'danger')
    
    return render_template('predict.html', prediction=prediction_result, 
                          model_predictions=model_predictions,
                          input_data=session.get('input_data', None))

@app.route('/crops')
def crops():
    return render_template('crops.html')

@app.route('/crop/<crop_name>')
def crop_detail(crop_name):
    return render_template('crop_detail.html', crop_name=crop_name)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)