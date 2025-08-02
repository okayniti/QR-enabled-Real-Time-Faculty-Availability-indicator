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

    # Enhanced, creative UI
    color = "#38d39f" if status == 1 else "#ff5353"
    emoji = "✅" if status == 1 else "⛔"
    msg = "1 - Available ✔️" if status == 1 else "0 - Not Available ❌"
    subtitle = "You can meet the faculty now!" if status == 1 else "Faculty is not available for in-person queries."
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Faculty Status</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                background: linear-gradient(135deg, #f7fafc 0%, #c3e1fc 100%);
                font-family: 'Segoe UI', Arial, sans-serif;
                text-align: center;
                padding: 40px;
                margin: 0;
            }}
            .status-card {{
                margin: 80px auto 20px auto;
                display: inline-block;
                background: white;
                border-radius: 24px;
                box-shadow: 0 4px 32px rgba(0,0,0,0.08);
                padding: 48px 40px;
            }}
            .msg {{
                font-size: 2.6rem;
                color: {color};
                margin: 16px 0 12px 0;
                font-weight: bold;
            }}
            .emoji {{
                font-size: 4.5rem;
                margin: 10px 0 20px 0;
            }}
            .sub {{
                font-size: 1rem;
                color: #72787d;
                margin-bottom: 30px;
            }}
            .faculty {{
                font-size: 1.25rem;
                color: #0474b8;
                margin-bottom: 24px;
            }}
            .timestamp {{
                font-size: 1rem;
                color: #B2B8BE;
                margin-top: 25px;
            }}
        </style>
    </head>
    <body>
        <div class="status-card">
            <div class="faculty">Faculty ID: {faculty_id}</div>
            <div class="emoji">{emoji}</div>
            <div class="msg">{msg}</div>
            <div class="sub">{subtitle}</div>
            <div class="timestamp">Updated at: {now.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </body>
    </html>
    '''
    return html_content

if __name__ == '__main__':
    app.run(debug=True)

