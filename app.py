import os
import pickle
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from models.crop_predictor import predict_crop

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
            
            # Make prediction using the enhanced ensemble prediction
            # Now returns both prediction and model predictions
            prediction_result, model_predictions = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            
            # Calculate vote counts for visualization
            vote_counts = {}
            for model_name, prediction in model_predictions.items():
                if prediction in vote_counts:
                    vote_counts[prediction] += 1
                else:
                    vote_counts[prediction] = 1
            
            flash(f'Recommended crop: {prediction_result}', 'success')
                
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            flash(f'Error making prediction: {str(e)}', 'danger')
    
    return render_template('predict.html', 
                          prediction=prediction_result, 
                          model_predictions=model_predictions,
                          vote_counts=vote_counts if 'vote_counts' in locals() else None,
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
