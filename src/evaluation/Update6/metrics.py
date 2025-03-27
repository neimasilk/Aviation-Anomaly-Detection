import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model
model = load_model('lstm_anomaly_detector.h5')

# Load and preprocess data
df = pd.read_csv('cleaned_data.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

# Prepare text features using TF-IDF with fixed number of features
vectorizer = TfidfVectorizer(max_features=332)
X = vectorizer.fit_transform(df['normalized_text']).toarray()

# Create target labels (assuming normal behavior)
y_true = np.zeros(len(df))  # Example: marking all instances as normal (0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM [samples, timesteps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Get model predictions
y_pred_proba = model.predict(X_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_proba)

# Hitung jumlah dan persentase anomali
jumlah_anomali = np.sum(y_pred)
persentase_anomali = (jumlah_anomali / len(y_pred)) * 100

# Membuat kesimpulan berdasarkan hasil metrik
kesimpulan = f"Model ini menunjukkan performa dengan akurasi {accuracy:.2%}. "
kesimpulan += f"Precision {precision:.2%} menunjukkan tingkat ketepatan prediksi positif, "
kesimpulan += f"sementara recall {recall:.2%} menunjukkan kemampuan model mendeteksi kasus positif. "
kesimpulan += f"F1-Score {f1:.2%} memberikan keseimbangan antara precision dan recall. "
kesimpulan += f"ROC-AUC {roc_auc:.2%} menunjukkan kemampuan model dalam membedakan kelas. "
kesimpulan += f"Model telah mendeteksi sebanyak {jumlah_anomali} anomali ({persentase_anomali:.2f}%)."

# Menyimpan hasil evaluasi ke CSV
hasil_metrics = pd.DataFrame({
    'Metrik': ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Jumlah Anomali', 'Persentase Anomali', 'Kesimpulan'],
    'Nilai': [f'{accuracy:.2%}', f'{precision:.2%}', f'{recall:.2%}', f'{f1:.2%}', f'{roc_auc:.2%}', 
             f'{jumlah_anomali}', f'{persentase_anomali:.2f}%', kesimpulan]
})

hasil_metrics.to_csv('hasil_metrics.csv', index=False)

# Print results
print('\nMetrik Evaluasi Model:')
print(f'Akurasi    : {accuracy:.2%}')
print(f'Precision  : {precision:.2%}')
print(f'Recall     : {recall:.2%}')
print(f'F1-Score   : {f1:.2%}')
print(f'ROC-AUC    : {roc_auc:.2%}')
print(f'Jumlah Anomali    : {jumlah_anomali}')
print(f'Persentase Anomali: {persentase_anomali:.2f}%\n')
print('Kesimpulan:')
print(kesimpulan)
print('\nHasil evaluasi telah disimpan dalam file hasil_metrics.csv')