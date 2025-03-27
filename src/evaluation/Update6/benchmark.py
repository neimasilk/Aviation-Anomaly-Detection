import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load data yang telah dibersihkan"""
    return pd.read_csv('cleaned_data.csv')

def manual_detection(data, threshold=0.5):
    """Simulasi deteksi anomali manual (sebagai baseline)"""
    # Menggunakan metode sederhana: menandai nilai yang melebihi threshold sebagai anomali
    start_time = time.time()
    # Hanya proses kolom numerik (START_TIME dan END_TIME)
    numeric_columns = ['START_TIME', 'END_TIME']
    anomalies = data[numeric_columns].apply(lambda x: abs(x - x.mean()) > (x.std() * threshold))
    end_time = time.time()
    
    processing_time = end_time - start_time
    num_anomalies = anomalies.sum().sum()
    return num_anomalies, processing_time

def lstm_detection(model, data):
    """Deteksi anomali menggunakan model LSTM"""
    start_time = time.time()
    # Preprocess data menggunakan TF-IDF
    vectorizer = TfidfVectorizer(max_features=332)
    X = vectorizer.fit_transform(data['normalized_text']).toarray()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data untuk LSTM [samples, timesteps, features]
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Prediksi menggunakan model
    predictions = model.predict(X_reshaped)
    end_time = time.time()
    
    processing_time = end_time - start_time
    num_anomalies = (predictions > 0.5).sum()
    return num_anomalies, processing_time

def main():
    # Load data dan model
    data = load_data()
    model = load_model('lstm_anomaly_detector.h5')
    
    # Benchmark metode manual
    manual_anomalies, manual_time = manual_detection(data)
    
    # Benchmark model LSTM
    lstm_anomalies, lstm_time = lstm_detection(model, data)
    
    # Hitung metrik performa
    total_samples = len(data)
    actual_anomalies = 177  # Berdasarkan hasil_metrics.csv
    
    # Simpan hasil benchmark
    benchmark_results = {
        'Metrik': [
            'Waktu Pemrosesan (Manual)',
            'Waktu Pemrosesan (LSTM)',
            'Jumlah Anomali Terdeteksi (Manual)',
            'Jumlah Anomali Terdeteksi (LSTM)',
            'Jumlah Anomali Aktual',
            'Persentase Deteksi (Manual)',
            'Persentase Deteksi (LSTM)',
            'Target Performa'
        ],
        'Nilai': [
            f'{manual_time:.3f} detik',
            f'{lstm_time:.3f} detik',
            manual_anomalies,
            lstm_anomalies,
            actual_anomalies,
            f'{(manual_anomalies/actual_anomalies)*100:.2f}%',
            f'{(lstm_anomalies/actual_anomalies)*100:.2f}%',
            'Deteksi 90% anomali dalam < 1 detik'
        ]
    }
    
    # Simpan hasil ke CSV
    pd.DataFrame(benchmark_results).to_csv('benchmark_results.csv', index=False)
    
    print('\nHasil Benchmark:')
    print('=' * 50)
    for metric, value in zip(benchmark_results['Metrik'], benchmark_results['Nilai']):
        print(f'{metric}: {value}')

if __name__ == '__main__':
    main()