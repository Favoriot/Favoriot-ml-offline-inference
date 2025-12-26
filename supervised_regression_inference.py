# ==================================================
# SUPERVISED LEARNING - REGRESSION INFERENCE
#
# Supported ML Algorithms:
#   - SVR (Support Vector Regression)
#   - Linear Regression
#   - Random Forest Regressor
#   - XGBoost Regressor
#
# Scaler Requirement:
#   - REQUIRED for:
#       * SVR
#       * Linear Regression
#   - NOT REQUIRED for:
#       * Random Forest Regressor
#       * XGBoost Regressor
#
# This script performs inference only (no training).
# ==================================================


# --------------------------------------------------
# Import required libraries
# --------------------------------------------------
import joblib              # Load trained ML artifacts
import pandas as pd        # Handle structured input data


# --------------------------------------------------
# 1. Load the trained Scaler (if required)
# --------------------------------------------------
# This scaler was fitted during training.
# It is required ONLY for:
#   - SVR
#   - Linear Regression
scaler = joblib.load('models/478433ec-dc3b-4049-a3a7-a29092cf7c58.joblib')


# --------------------------------------------------
# 2. Load the trained supervised regression model
# --------------------------------------------------
# The model file stores:
#   - "regressor" → trained regression model
with open('models/478433ec-dc3b-4049-a3a7-a29092cf7c58.pkl', 'rb') as f:
    model = joblib.load(f)


# --------------------------------------------------
# 3. Prepare raw input data
# --------------------------------------------------
# Feature names and order MUST match training data
raw_data = pd.DataFrame(
    [[80, 27]],          # Example sensor values
    columns=['lux', 'temp']
)


# --------------------------------------------------
# 4. Apply feature scaling (ONLY if required)
# --------------------------------------------------
# Scaling is required for:
#   - SVR
#   - Linear Regression
#
# Random Forest and XGBoost do NOT require scaling.
scaled_data = scaler.transform(raw_data)



# --------------------------------------------------
# 5. Run supervised regression inference
# --------------------------------------------------
# The model outputs a continuous numeric value
prediction = model.predict(scaled_data)
# Example output: [42.73]


# --------------------------------------------------
# 6. Output final regression result
# --------------------------------------------------
print(f"Supervised Regression Result: {prediction[0]}")
