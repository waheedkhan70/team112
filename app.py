from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the pre-trained model and feature engineer
try:
    with open("logistic_fraud_detector.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_engineer.pkl", "rb") as f:
        engineer = pickle.load(f)
    logger.info("Model and feature engineer loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model or feature engineer file not found: {e}")
    raise Exception("Model or feature engineer file not found. Please ensure 'logistic_fraud_detector.pkl' and 'feature_engineer.pkl' are in the project directory.")
except Exception as e:
    logger.error(f"Error loading model or feature engineer: {e}")
    raise Exception(f"Error loading model or feature engineer: {e}")

def process_transaction(tx_data):
    try:
        logger.debug(f"Received transaction data: {tx_data}")
        df = pd.DataFrame([tx_data])

        # Parse date and time
        try:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
            df['Transaction Time'] = pd.to_datetime(df['Transaction Time'], format='%H:%M', errors='coerce')
            if df['Transaction Date'].isna().any() or df['Transaction Time'].isna().any():
                return None, "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."
        except Exception as e:
            logger.error(f"Date/time parsing error: {e}")
            return None, "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."

        # Extract datetime features
        df['Trans_Day'] = df['Transaction Date'].dt.day
        df['Trans_Month'] = df['Transaction Date'].dt.month
        df['Trans_Weekday'] = df['Transaction Date'].dt.weekday
        df['Trans_Hour'] = df['Transaction Time'].dt.hour

        # Extract city and state
        df[['Sender_City', 'Sender_State']] = df['Sender Location'].str.extract(r'([^,]+),\s*([^,]+)')
        df[['Receiver_City', 'Receiver_State']] = df['Receiver Location'].str.extract(r'([^,]+),\s*([^,]+)')

        # Check for invalid location data
        if df[['Sender_City', 'Sender_State', 'Receiver_City', 'Receiver_State']].isna().any().any():
            return None, "Invalid location format. Please use 'City, State' (e.g., 'Kolkata, West Bengal')."

        # Drop original columns
        df.drop(columns=['Transaction Date', 'Transaction Time', 'Sender Location', 'Receiver Location'], inplace=True)

        # Log DataFrame before transformation
        logger.debug(f"DataFrame before feature engineering: {df.to_dict()}")

        # Apply feature engineering
        try:
            df_transformed = engineer.transform(df)
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return None, f"Feature engineering failed: {e}"

        # Ensure numeric types
        df_transformed = df_transformed.apply(pd.to_numeric, errors='coerce')
        df_transformed.fillna(0, inplace=True)

        # Log transformed DataFrame
        logger.debug(f"Transformed DataFrame: {df_transformed.to_dict()}")

        # Verify feature names
        expected_features = getattr(engineer, 'feature_names_out', None) or model.feature_names_in_
        if set(df_transformed.columns) != set(expected_features):
            logger.error(f"Feature mismatch. Expected: {expected_features}, Got: {df_transformed.columns.tolist()}")
            return None, "Feature names do not match the model's expected features."

        return df_transformed, None
    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        return None, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        tx_data = {
            'Amount (INR)': float(request.form['amount']),
            'Transaction Date': request.form['tx_date'],
            'Transaction Time': request.form['tx_time'],
            'Sender Location': request.form['sender_loc'],
            'Receiver Location': request.form['receiver_loc'],
            'Payment Mode': request.form['payment_mode'],
            'To/Received By': request.form['receiver'],
            'Sender A/C': request.form['sender_acc'],
            'Receiver A/C': request.form['receiver_acc']
        }
        logger.debug(f"Form data received: {tx_data}")

        # Process input
        processed_df, error = process_transaction(tx_data)  # Fixed: Only two values to unpack
        if error:
            logger.error(f"Processing error: {error}")
            return jsonify({'error': error}), 400

        # Make prediction
        try:
            prediction = model.predict(processed_df)[0]
            probability = model.predict_proba(processed_df)[0][1]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': f"Prediction failed: {e}"}), 400

        result = {
            'prediction': 'Fraud Detected' if prediction == 1 else 'Legitimate Transaction',
            'probability': round(probability * 100, 2),
            'status': 'danger' if prediction == 1 else 'success'
        }
        logger.info(f"Prediction result: {result}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)