import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load hasil metrics
hasil_metrics = pd.read_csv('hasil_metrics.csv')

# Load data dan prediksi dari metrics.py
df = pd.read_csv('cleaned_data.csv')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Persiapkan data
vectorizer = TfidfVectorizer(max_features=332)
X = vectorizer.fit_transform(df['normalized_text']).toarray()
y_true = np.zeros(len(df))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Load model dan dapatkan prediksi
model = load_model('lstm_anomaly_detector.h5')
y_pred_proba = model.predict(X_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 1. Grafik Bar Perbandingan Akurasi
plt.figure(figsize=(10, 6))
metrics_data = hasil_metrics[hasil_metrics['Metrik'].isin(['Akurasi', 'Precision', 'Recall', 'F1-Score'])]
metrics_values = [float(val.strip('%'))/100 for val in metrics_data['Nilai']]

plt.bar(metrics_data['Metrik'], metrics_values, color=['blue', 'green', 'red', 'purple'])
plt.title('Perbandingan Metrik Evaluasi Model')
plt.ylabel('Nilai')
plt.ylim(0, 1)

# Tambahkan nilai di atas bar
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')

plt.savefig('plots/metrics_comparison.png')
plt.close()

# 2. Heatmap Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# 3. Kurva ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('plots/roc_curve.png')
plt.close()

print('Visualisasi telah dibuat dan disimpan dalam folder plots sebagai:')
print('1. plots/metrics_comparison.png - Grafik bar perbandingan metrik')
print('2. plots/confusion_matrix.png - Heatmap confusion matrix')
print('3. plots/roc_curve.png - Kurva ROC')