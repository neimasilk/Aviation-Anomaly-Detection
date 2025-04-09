import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
import h5py
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_model_and_data():
    # Load data
    data = pd.read_csv('cleaned_data.csv')
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    X = data[numeric_cols].values  # Gunakan numpy array untuk menghindari masalah feature names
    
    # Load model parameters
    with h5py.File('Isolation_Forest.h5', 'r') as h5file:
        model_params = eval(h5file['model_params'][()])
    
    return X, model_params

def cross_validation(X, model_params, n_splits=5):
    print("\nMelakukan Validasi Silang (K-Fold Cross Validation)...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Train data dan validation data
        X_train = X[train_idx]  # Gunakan numpy array indexing
        X_val = X[val_idx]
        
        # Train model
        model = IsolationForest(**model_params)
        model.fit(X_train)
        
        # Prediksi
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Hitung anomali rate
        train_anomaly_rate = np.mean(train_pred == -1)
        val_anomaly_rate = np.mean(val_pred == -1)
        
        print(f"\nFold {fold}:")
        print(f"Training Anomaly Rate: {train_anomaly_rate:.2%}")
        print(f"Validation Anomaly Rate: {val_anomaly_rate:.2%}")
        
        scores.append({
            'train_anomaly_rate': train_anomaly_rate,
            'val_anomaly_rate': val_anomaly_rate
        })
    
    # Rata-rata hasil
    avg_train_rate = np.mean([s['train_anomaly_rate'] for s in scores])
    avg_val_rate = np.mean([s['val_anomaly_rate'] for s in scores])
    print(f"\nRata-rata Anomaly Rate:")
    print(f"Training: {avg_train_rate:.2%}")
    print(f"Validation: {avg_val_rate:.2%}")

def offline_evaluation(X, model_params):
    print("\nMelakukan Evaluasi Offline...")
    
    # Bagi data menjadi data historis (80%) dan data test (20%)
    train_size = int(0.8 * len(X))
    X_historical = X[:train_size]  # Gunakan numpy array slicing
    X_test = X[train_size:]
    
    # Train model dengan data historis
    model = IsolationForest(**model_params)
    model.fit(X_historical)
    
    # Evaluasi pada data test
    predictions = model.predict(X_test)
    anomaly_rate = np.mean(predictions == -1)
    
    print(f"Jumlah data historis: {len(X_historical)}")
    print(f"Jumlah data test: {len(X_test)}")
    print(f"Anomaly rate pada data test: {anomaly_rate:.2%}")

def realtime_evaluation(X, model_params, sample_size=1000):
    print("\nMelakukan Evaluasi Real-Time...")
    
    # Train model
    model = IsolationForest(**model_params)
    model.fit(X)
    
    # Ambil sampel data untuk simulasi real-time
    indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
    sample_data = X[indices]  # Gunakan numpy array indexing
    
    # Ukur waktu pemrosesan
    processing_times = []
    for data_point in sample_data:
        start_time = time.time()
        _ = model.predict(data_point.reshape(1, -1))
        end_time = time.time()
        processing_times.append(end_time - start_time)
    
    avg_time = np.mean(processing_times)
    max_time = np.max(processing_times)
    min_time = np.min(processing_times)
    
    print(f"Rata-rata waktu pemrosesan: {avg_time*1000:.2f} ms")
    print(f"Waktu pemrosesan maksimum: {max_time*1000:.2f} ms")
    print(f"Waktu pemrosesan minimum: {min_time*1000:.2f} ms")

def main():
    print("Memulai validasi model Isolation Forest...")
    X, model_params = load_model_and_data()
    
    # Jalankan semua validasi
    cross_validation(X, model_params)
    offline_evaluation(X, model_params)
    realtime_evaluation(X, model_params)
    
    print("\nValidasi selesai!")

if __name__ == "__main__":
    main()