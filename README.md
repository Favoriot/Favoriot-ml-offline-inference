# Favoriot ML Offline Inference Deployment

This repository contains **pre-trained Machine Learning models** and inference scripts for **offline deployment** on IoT devices or gateway systems.  
The models were **trained using Favoriot platform ML features** and are ready for **real-time edge inference** without requiring cloud connectivity.

> **Note:** The models were trained in **Python 3.12++**. Ensure your deployment environment uses Python 3.12 or higher.


---

## Supported ML Features (Favoriot Platform)

### Supervised Classification
- Algorithms: SVM, Logistic Regression, XGBoost, Random Forest  
- Scaler required: SVM, Logistic Regression, XGBoost  

### Supervised Regression
- Algorithms: SVM/SVR, Linear Regression, Random Forest, XGBoost  
- Scaler required: SVR, Linear Regression  

### Unsupervised Clustering
- Algorithms: K-Means, DBSCAN, Gaussian Mixture Model (GMM)  
- Scaler required for all  
- DBSCAN uses a **custom label assignment function** for new offline data

> All models were trained using **Favoriot platform ML features**, including automated preprocessing, feature selection, and model tuning.

---

## 🛠 Setup Instructions for Offline Inference

1. **Clone the repository**  

```bash
git clone https://github.com/username/favoriot-ml-offline.git
cd favoriot-ml-offline
```
2. **Create a virtual environment
```bash
python3.12 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Place ML model and scalar 
- .joblib scaler and .pkl model must be in the models/ folder
5. Running Offline Inference
```bash
python supervised_classification_inference.py
python supervised_regression_inference.py
python unsupervised_clustering_inference.py
```
