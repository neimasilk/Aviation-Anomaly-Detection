import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from joblib import load

# Load model dan scaler yang telah disimpan
model = load('one_class_svm.h5')
scaler = load('scaler.h5')

# Baca data
df = pd.read_csv('cleaned_data.csv')

# Pilih fitur yang sama dengan training
features = ['START_TIME', 'END_TIME']
X = df[features]

# Preprocessing data
X_scaled = scaler.transform(X)

# Lakukan prediksi
# One-Class SVM mengembalikan +1 untuk normal dan -1 untuk anomali
# Kita ubah menjadi 0 untuk normal dan 1 untuk anomali agar sesuai dengan format metrik evaluasi
y_pred = model.predict(X_scaled)
y_pred = np.where(y_pred == 1, 0, 1)  # Konversi +1 -> 0 (normal) dan -1 -> 1 (anomali)

# Untuk keperluan evaluasi, kita asumsikan 10% data adalah anomali (sesuai dengan parameter nu=0.1)
y_true = np.zeros(len(y_pred))
y_true[np.argsort(model.score_samples(X_scaled))[:int(len(y_pred) * 0.1)]] = 1

# Hitung metrik-metrik evaluasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# Tampilkan hasil evaluasi
print('Hasil Evaluasi Model One-Class SVM:')
print(f'Akurasi   : {accuracy:.2%}')
print(f'Precision : {precision:.2%}')
print(f'Recall    : {recall:.2%}')
print(f'F1-Score  : {f1:.2%}')
print(f'ROC-AUC   : {roc_auc:.2%}')