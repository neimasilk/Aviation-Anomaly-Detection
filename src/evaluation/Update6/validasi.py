import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Load data
df = pd.read_csv('cleaned_data.csv')

# Prepare features
vectorizer = TfidfVectorizer(max_features=332)
X = vectorizer.fit_transform(df['normalized_text']).toarray()
y = np.zeros(len(df))  # Assuming all data is normal

# Load model
model = load_model('lstm_anomaly_detector.h5')

def cross_validation(X, y, n_splits=5):
    """Melakukan K-Fold Cross Validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Reshape for LSTM
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        
        # Get predictions
        y_pred_proba = model.predict(X_val_reshaped)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        fold_metrics = {
            'fold': fold,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        metrics.append(fold_metrics)
    
    return pd.DataFrame(metrics)

def offline_validation(X, y):
    """Melakukan validasi offline menggunakan data historis"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Get predictions
    y_pred_proba = model.predict(X_reshaped)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }
    
    return metrics

def real_time_validation(text_sample):
    """Melakukan validasi real-time untuk mengukur kecepatan pemrosesan"""
    start_time = time.time()
    
    # Preprocess single sample
    X_sample = vectorizer.transform([text_sample]).toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    
    # Get prediction
    prediction = model.predict(X_reshaped)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        'prediction': bool(prediction > 0.5),
        'processing_time': processing_time
    }

# Jalankan validasi
print('\nMenjalankan K-Fold Cross Validation...')
cv_results = cross_validation(X, y)
print('\nHasil Cross Validation:')
print(cv_results)

# Hitung rata-rata metrik Cross Validation
cv_mean = cv_results.mean()[['accuracy', 'precision', 'recall', 'f1']]
print('\nRata-rata metrik Cross Validation:')
print(cv_mean)

print('\nMenjalankan Validasi Offline...')
offline_results = offline_validation(X, y)
print('\nHasil Validasi Offline:')
for metric, value in offline_results.items():
    print(f'{metric}: {value:.4f}')

# Contoh validasi real-time
sample_text = df['normalized_text'].iloc[0]
print('\nMenjalankan Validasi Real-time...')
rt_result = real_time_validation(sample_text)
print('\nHasil Validasi Real-time:')
print(f'Waktu pemrosesan: {rt_result["processing_time"]:.4f} detik')
print(f'Prediksi: {"Anomali" if rt_result["prediction"] else "Normal"}')

# Menyimpan semua hasil validasi ke CSV
hasil_validasi = pd.DataFrame()

# Simpan hasil Cross Validation
cv_results['Jenis_Validasi'] = 'Cross Validation'
hasil_validasi = pd.concat([hasil_validasi, cv_results])

# Tambahkan rata-rata Cross Validation
cv_mean_df = pd.DataFrame([{
    'fold': 'Rata-rata',
    'accuracy': cv_mean['accuracy'],
    'precision': cv_mean['precision'],
    'recall': cv_mean['recall'],
    'f1': cv_mean['f1'],
    'Jenis_Validasi': 'Cross Validation (Rata-rata)'
}])
hasil_validasi = pd.concat([hasil_validasi, cv_mean_df])

# Simpan hasil Validasi Offline
offline_df = pd.DataFrame([{
    'fold': 'Offline',
    **offline_results,
    'Jenis_Validasi': 'Validasi Offline'
}])
hasil_validasi = pd.concat([hasil_validasi, offline_df])

# Simpan hasil Validasi Real-time
rt_df = pd.DataFrame([{
    'fold': 'Real-time',
    'processing_time': rt_result['processing_time'],
    'prediction': 'Anomali' if rt_result['prediction'] else 'Normal',
    'Jenis_Validasi': 'Validasi Real-time'
}])
hasil_validasi = pd.concat([hasil_validasi, rt_df])

# Simpan ke file CSV
hasil_validasi.to_csv('hasil_validasi.csv', index=False)
print('\nHasil validasi telah disimpan dalam file hasil_validasi.csv')