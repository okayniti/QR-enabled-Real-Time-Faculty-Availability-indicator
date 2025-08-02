import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the data
data = pd.read_csv('faculty_data.csv')

# 2. Prepare features and label
X = data[['DayOfWeek', 'TimeSlot', 'ScheduledClass', 'HasMeeting', 'PrevAvailable']].values
y = data['Status'].values

# 3. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Build the model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=2)

# 7. Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {acc:.2f}')

# 8. Save scaler and model
joblib.dump(scaler, 'scaler.save')
model.save('ann_faculty_model.keras')  # recommended native keras format

