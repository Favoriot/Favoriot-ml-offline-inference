# Favoriot ML Offline Inference Deployment

This repository contains **pre-trained Machine Learning models** and inference scripts for **offline deployment** on IoT devices or gateway systems.  
The models were **trained using Favoriot platform ML features** and are ready for **real-time edge inference** without requiring cloud connectivity.

> **Note:** The models were trained in **Python 3.12++**. Ensure your deployment environment uses Python 3.12 or higher.

---

## 📂 Repository Structure

├── models/
│ ├── <scaler>.joblib # StandardScaler from Favoriot training
│ └── <model>.pkl # ML model trained on Favoriot platform
│
├── supervised_classification_inference.py # Supervised classification
├── supervised_regression_inference.py # Supervised regression
└── unsupervised_clustering_inference.py # Unsupervised clustering
│
├── README.md
└── .gitignore
