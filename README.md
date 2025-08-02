# Faculty QR ANN Availability Indicator

This project predicts real-time faculty availability using an Artificial Neural Network, Flask for the web app, and QR codes for quick access.

## Setup
1. Install requirements with: pip install -r requirements.txt
2. Run train_model.py to generate the model and scaler files.
3. Start the Flask app with: python app.py
4. Generate QR codes for each faculty using generate_all_qr.py

## Usage
Scan the faculty QR codes to get real-time availability (0 - Not Available, 1 - Available) in your browser.

## Requirements
- Python 3.x
- pandas, tensorflow, scikit-learn, flask, qrcode, joblib

## Sample
![Faculty 1 Sample QR Output](faculty_1_qr.png)
