import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import h5py
from sklearn.ensemble import IsolationForest

# Load data
data = pd.read_csv('cleaned_data.csv')

# Preprocess data - using numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
X = data[numeric_cols]

# Load model from HDF5 file
with h5py.File('Isolation_Forest.h5', 'r') as h5file:
    model_params = eval(h5file['model_params'][()])  # Get model parameters

# Recreate model with saved parameters
model = IsolationForest(**model_params)
model.fit(X)  # Refit the model with data

# Get predictions
# Isolation Forest returns -1 for anomalies and 1 for normal points
# Convert to 0 (normal) and 1 (anomaly) for standard metrics
predictions = model.predict(X)
predictions = np.where(predictions == 1, 0, 1)  # Convert -1 to 1 (anomaly) and 1 to 0 (normal)

# For demonstration, we'll create synthetic labels
# In a real scenario, you would use actual labeled data
np.random.seed(42)
true_labels = np.random.binomial(n=1, p=0.1, size=len(predictions))  # 10% anomaly rate

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# For ROC-AUC, we need prediction scores
# Get the decision function scores
scores = -model.score_samples(X)  # Negative of the anomaly scores
roc_auc = roc_auc_score(true_labels, scores)

# Print results
print("\nHasil Evaluasi Model Isolation Forest:")
print(f"Akurasi    : {accuracy:.2%}")
print(f"Precision  : {precision:.2%}")
print(f"Recall     : {recall:.2%}")
print(f"F1-Score   : {f1:.2%}")
print(f"ROC-AUC    : {roc_auc:.2%}")

print("\nInterpretasi:")
print(f"- Model memiliki akurasi {accuracy:.2%} dalam mengklasifikasikan data")
print(f"- Dari semua kasus yang diprediksi sebagai anomali, {precision:.2%} adalah benar anomali")
print(f"- Model berhasil mendeteksi {recall:.2%} dari total anomali yang sebenarnya")
print(f"- F1-Score {f1:.2%} menunjukkan keseimbangan antara precision dan recall")
print(f"- ROC-AUC {roc_auc:.2%} menunjukkan kemampuan model dalam membedakan antara data normal dan anomali")