import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import h5py
from sklearn.ensemble import IsolationForest

# Set style plotting
plt.style.use('default')

# Set figure style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.3

# Load data dan model
data = pd.read_csv('cleaned_data.csv')
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
X = data[numeric_cols]

# Load model
with h5py.File('Isolation_Forest.h5', 'r') as h5file:
    model_params = eval(h5file['model_params'][()])

# Recreate model
model = IsolationForest(**model_params)
model.fit(X)

# Get predictions and scores
predictions = model.predict(X)
predictions = np.where(predictions == 1, 0, 1)  # Convert -1 to 1 (anomaly)
scores = -model.score_samples(X)  # Negative of the anomaly scores

# Create synthetic labels for demonstration
np.random.seed(42)
true_labels = np.random.binomial(n=1, p=0.1, size=len(predictions))

# Membuat figure dengan 3 subplot
plt.figure(figsize=(18, 6))

# 1. Grafik Bar - Perbandingan Akurasi
plt.subplot(131)
accuracy_data = {
    'Sebelum': 0.85,  # Contoh nilai
    'Sesudah': 0.92   # Contoh nilai
}
bar_colors = ['#2ecc71', '#3498db']
plt.bar(accuracy_data.keys(), accuracy_data.values(), color=bar_colors)
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Akurasi')
plt.ylim(0, 1)
for i, v in enumerate(accuracy_data.values()):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')

# 2. Heatmap - Area Kesalahan Prediksi
plt.subplot(132)
error_matrix = np.zeros((2, 2))
error_matrix[0, 0] = np.sum((predictions == 0) & (true_labels == 0))  # True Negatives
error_matrix[0, 1] = np.sum((predictions == 1) & (true_labels == 0))  # False Positives
error_matrix[1, 0] = np.sum((predictions == 0) & (true_labels == 1))  # False Negatives
error_matrix[1, 1] = np.sum((predictions == 1) & (true_labels == 1))  # True Positives

sns.heatmap(error_matrix, annot=True, fmt='g', cmap='RdYlBu_r',
            xticklabels=['Normal', 'Anomali'],
            yticklabels=['Normal', 'Anomali'])
plt.title('Heatmap Prediksi vs Aktual')

# 3. Kurva ROC
plt.subplot(133)
fpr, tpr, _ = roc_curve(true_labels, scores)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
plt.legend(loc='lower right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print('Visualisasi telah disimpan sebagai "model_visualization.png"')