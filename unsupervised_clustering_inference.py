# ==================================================
# UNSUPERVISED LEARNING - CLUSTERING INFERENCE
#
# Supported Algorithms:
#   - K-Means
#   - DBSCAN
#   - GMM (Gaussian Mixture Model)
#
# Scaler Requirement:
#   - REQUIRED for:
#       * K-Means
#       * DBSCAN
#       * GMM
#
# Notes:
#   - K-Means & GMM support direct .predict()
#   - DBSCAN does NOT support .predict()
#     → Custom label assignment is required
# ==================================================


# --------------------------------------------------
# Import required libraries
# --------------------------------------------------
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


# --------------------------------------------------
# 1. Load trained scaler
# --------------------------------------------------
# Scaler is required for ALL supported clustering algorithms
# because distance-based calculations are used.
scaler = joblib.load('models/9c822717-9364-40bc-8177-7cf08f1d0683.joblib')


# --------------------------------------------------
# 2. Load trained clustering model
# --------------------------------------------------
# The model file contains one of:
#   - KMeans
#   - DBSCAN
#   - GaussianMixture
with open('models/9c822717-9364-40bc-8177-7cf08f1d0683.pkl', 'rb') as f:
    model = joblib.load(f)


# --------------------------------------------------
# 3. Prepare raw input data
# --------------------------------------------------
# Feature names and order MUST match training data
raw_data = pd.DataFrame(
    [[60.0,80,25.5]],
    columns=['humi','lux','temp']
)


# --------------------------------------------------
# 4. Apply feature scaling (MANDATORY)
# --------------------------------------------------
scaled_data = scaler.transform(raw_data)


# --------------------------------------------------
# 5. DBSCAN custom label assignment function
# --------------------------------------------------
# DBSCAN does not support .predict()
# This function assigns cluster labels to new data
# based on distance to core samples.
def dbscanAssignLabel(model, new_data):
    core_samples = model.components_
    distances = pairwise_distances(new_data, core_samples)

    # Default label = -1 (noise)
    labels = np.full(new_data.shape[0], -1)

    for i, dist in enumerate(distances):
        if np.min(dist) <= model.eps:
            labels[i] = model.labels_[np.argmin(dist)]

    return labels


# --------------------------------------------------
# 6. Run unsupervised clustering inference
# --------------------------------------------------
# K-Means & GMM → direct prediction
# DBSCAN → custom label assignment
if hasattr(model, "predict"):
    # K-Means or GMM
    prediction = model.predict(scaled_data)
else:
    # DBSCAN
    prediction = dbscanAssignLabel(model, scaled_data)


# --------------------------------------------------
# 7. Output clustering result
# --------------------------------------------------
print(f"Unsupervised Clustering Result (Cluster ID): {prediction[0]}")
