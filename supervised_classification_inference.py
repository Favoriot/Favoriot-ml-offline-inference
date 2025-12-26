import joblib
import pandas as pd

# 1. Load the Scaler (Joblib file)
# This file contains the 'StandardScaler' to normalize your input
scaler = joblib.load('models/1d0640b0-5715-4e06-986c-770ca2505dad.joblib')

# 2. Load the Model (Pkl file)
# This is your XGBoost classifier brain
with open('models/1d0640b0-5715-4e06-986c-770ca2505dad.pkl', 'rb') as f:
    model = joblib.load(f)

# 3. Prepare raw sensor data
raw_data = pd.DataFrame([[25.5, 60.0]], columns=['temp', 'humi']) # replace with you input x in set in training model

# 4. Apply the Scaler BEFORE predicting
# This converts raw numbers (e.g. 25.5) into the scaled format the model expects
scaled_data = scaler.transform(raw_data)

classifier = model["classifier"]
label_encoder = model.get("label_encoder", None)


# 5. Run Inference
prediction = classifier.predict(scaled_data)
if label_encoder:
    output = label_encoder.inverse_transform(prediction)
    
print(f"Prediction result: {output[0]}")