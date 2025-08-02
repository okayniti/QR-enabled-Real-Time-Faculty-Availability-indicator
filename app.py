from flask import Flask
import datetime
import numpy as np
import joblib
from tensorflow import keras

app = Flask(__name__)

scaler = joblib.load('scaler.save')
model = keras.models.load_model('ann_faculty_model.keras')

def predict_availability(day, hour, scheduled_class, has_meeting, prev_available):
    X = np.array([[day, hour, scheduled_class, has_meeting, prev_available]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0][0]
    return 1 if pred > 0.5 else 0

@app.route('/faculty/<int:faculty_id>')
def faculty_status(faculty_id):
    import datetime
    now = datetime.datetime.now()
    day = now.weekday()
    hour = now.hour
    scheduled_class = 0
    has_meeting = 0
    prev_available = 1
    status = predict_availability(day, hour, scheduled_class, has_meeting, prev_available)

    # Improved UI HTML content
    html_content = f'''
    <html>
    <head>
        <title>Faculty Availability Status</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f6f7;
                color: #333;
                text-align: center;
                padding: 50px;
            }}
            .status-box {{
                display: inline-block;
                padding: 40px 60px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                background-color: {'#28a745' if status == 1 else '#dc3545'};
                color: white;
                font-size: 48px;
                font-weight: bold;
            }}
            .faculty-info {{
                font-size: 32px;
                margin-bottom: 20px;
            }}
            .footer {{
                margin-top: 40px;
                font-size: 16px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="faculty-info">Faculty ID: {faculty_id}</div>
        <div class="status-box">{status} - {'Available' if status == 1 else 'Not Available'}</div>
        <div class="footer">Updated at: {now.strftime('%Y-%m-%d %H:%M:%S')}</div>
    </body>
    </html>
    '''

    return html_content

if __name__ == '__main__':
    app.run(debug=True)
