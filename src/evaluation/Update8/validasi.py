import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from joblib import load
import time

def validasi_silang(data, features, n_splits=5):
    print('\nMelakukan Validasi Silang (K-Fold Cross Validation):')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        # Persiapkan data
        X_train = data.iloc[train_idx][features]
        X_val = data.iloc[val_idx][features]
        
        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Latih model
        model = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        model.fit(X_train_scaled)
        
        # Evaluasi
        val_score = model.score_samples(X_val_scaled)
        anomaly_ratio = np.mean(val_score < model.offset_)
        scores.append(anomaly_ratio)
        print(f'Fold {fold}: Rasio Anomali = {anomaly_ratio:.2%}')
    
    print(f'Rata-rata Rasio Anomali: {np.mean(scores):.2%}')
    print(f'Standar Deviasi: {np.std(scores):.2%}')

def uji_offline(model, scaler, data, features):
    print('\nMelakukan Uji Offline dengan Data Historis:')
    # Preprocessing data
    X = data[features]
    X_scaled = scaler.transform(X)
    
    # Prediksi
    start_time = time.time()
    predictions = model.predict(X_scaled)
    end_time = time.time()
    
    # Hitung metrik
    anomaly_ratio = np.mean(predictions == -1)
    processing_time = end_time - start_time
    
    print(f'Jumlah data yang diuji: {len(data)}')
    print(f'Rasio anomali terdeteksi: {anomaly_ratio:.2%}')
    print(f'Waktu pemrosesan total: {processing_time:.2f} detik')
    print(f'Rata-rata waktu per data: {(processing_time/len(data))*1000:.2f} ms')

def uji_real_time(model, scaler, data, features, batch_size=1):
    print('\nMelakukan Uji Real-Time:')
    X = data[features].iloc[:batch_size]
    X_scaled = scaler.transform(X)
    
    processing_times = []
    for _ in range(10):  # Lakukan 10 kali pengujian
        start_time = time.time()
        model.predict(X_scaled)
        end_time = time.time()
        processing_times.append(end_time - start_time)
    
    avg_time = np.mean(processing_times) * 1000  # Konversi ke milidetik
    std_time = np.std(processing_times) * 1000
    
    print(f'Rata-rata waktu pemrosesan: {avg_time:.2f} ms')
    print(f'Standar deviasi: {std_time:.2f} ms')
    print(f'Waktu pemrosesan minimum: {min(processing_times)*1000:.2f} ms')
    print(f'Waktu pemrosesan maksimum: {max(processing_times)*1000:.2f} ms')

def main():
    # Load data dan model
    print('Loading data dan model...')
    df = pd.read_csv('cleaned_data.csv')
    model = load('one_class_svm.h5')
    scaler = load('scaler.h5')
    features = ['START_TIME', 'END_TIME']
    
    # Jalankan semua validasi
    validasi_silang(df, features)
    uji_offline(model, scaler, df, features)
    uji_real_time(model, scaler, df, features)

if __name__ == '__main__':
    main()