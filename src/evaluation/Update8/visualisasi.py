import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from joblib import load

# Load model dan scaler
model = load('one_class_svm.h5')
scaler = load('scaler.h5')

# Baca data
df = pd.read_csv('cleaned_data.csv')
features = ['START_TIME', 'END_TIME']
X = df[features]

# Preprocessing data
X_scaled = scaler.transform(X)

# Lakukan prediksi
y_pred = model.predict(X_scaled)
y_pred = np.where(y_pred == 1, 0, 1)  # Konversi +1 -> 0 (normal) dan -1 -> 1 (anomali)

# Ground truth (10% data adalah anomali)
y_true = np.zeros(len(y_pred))
y_true[np.argsort(model.score_samples(X_scaled))[:int(len(y_pred) * 0.1)]] = 1

# Buat figure dengan 3 subplot
plt.style.use('default')
fig = plt.figure(figsize=(15, 5))

# Set warna dan style yang menarik
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 1. Grafik Bar - Perbandingan Akurasi
plt.subplot(131)
metrics = {
    'Normal': np.mean(y_pred[y_true == 0] == 0),
    'Anomali': np.mean(y_pred[y_true == 1] == 1)
}
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title('Akurasi Model per Kelas')
plt.ylabel('Akurasi')

# 2. Heatmap - Area Kesalahan Prediksi
plt.subplot(132)
confusion_matrix = pd.crosstab(y_true, y_pred, normalize='index')
sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='YlOrRd')
plt.title('Heatmap Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')

# 3. Kurva ROC
plt.subplot(133)
fpr, tpr, _ = roc_curve(y_true, model.score_samples(X_scaled))
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Atur layout dan tampilkan
plt.tight_layout()
plt.savefig('visualisasi_hasil.png', dpi=300, bbox_inches='tight')
plt.show()