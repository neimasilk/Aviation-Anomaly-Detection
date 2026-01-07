# Model_Isolation_forest.py
# Implementasi model Isolation Forest untuk deteksi anomali dalam komunikasi penerbangan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
import os
import joblib
from pathlib import Path
from tensorflow import keras
import pickle
import h5py

# Konfigurasi visualisasi
plt.style.use('fivethirtyeight')
sns.set(style='whitegrid')

class AnomalyDetector:
    def __init__(self, random_state=42, contamination=0.05, n_estimators=100, max_samples='auto'):
        """
        Inisialisasi model Isolation Forest untuk deteksi anomali
        
        Parameters:
        -----------
        random_state : int, default=42
            Seed untuk reproducibility
        contamination : float, default=0.05
            Proporsi anomali yang diharapkan dalam dataset
        n_estimators : int, default=100
            Jumlah pohon dalam ensemble
        max_samples : int or str, default='auto'
            Jumlah sampel untuk membangun setiap pohon
        """
        self.random_state = random_state
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.model = IsolationForest(
            random_state=self.random_state,
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_jobs=-1  # Menggunakan semua core yang tersedia
        )
        self.model_path = None
    
    def load_data(self, file_path):
        """
        Memuat data dari file CSV
        
        Parameters:
        -----------
        file_path : str
            Path ke file CSV yang berisi data yang sudah dibersihkan
            
        Returns:
        --------
        pandas.DataFrame
            Data yang dimuat dari file CSV
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Data berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom")
            print(f"Kolom dalam dataset: {data.columns.tolist()}")
            return data
        except Exception as e:
            print(f"Error saat memuat data: {e}")
            return None
    
    def preprocess_data(self, data):
        """
        Melakukan preprocessing pada data sebelum training
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data yang akan dipreprocessing
            
        Returns:
        --------
        pandas.DataFrame
            Data yang sudah dipreprocessing
        """
        # Cek missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values ditemukan:\n{missing_values[missing_values > 0]}")
            # Isi missing values dengan median untuk kolom numerik
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
        
        # Hanya pilih kolom numerik untuk model Isolation Forest
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        print(f"Menggunakan {numeric_data.shape[1]} fitur numerik untuk deteksi anomali")
        
        return numeric_data
    
    def train(self, data, test_size=0.2):
        """
        Melatih model Isolation Forest
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data yang akan digunakan untuk melatih model
        test_size : float, default=0.2
            Proporsi data yang akan digunakan sebagai test set
            
        Returns:
        --------
        tuple
            (X_train, X_test) - Data training dan testing
        """
        # Split data menjadi training dan testing
        X_train, X_test = train_test_split(data, test_size=test_size, random_state=self.random_state)
        print(f"Data dibagi menjadi {X_train.shape[0]} sampel training dan {X_test.shape[0]} sampel testing")
        
        # Fit model pada data training
        print("Melatih model Isolation Forest...")
        self.model.fit(X_train)
        print("Model berhasil dilatih!")
        
        return X_train, X_test
    
    def predict(self, data):
        """
        Memprediksi anomali pada data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data yang akan diprediksi
            
        Returns:
        --------
        numpy.ndarray
            Array berisi label anomali (1: normal, -1: anomali)
        """
        predictions = self.model.predict(data)
        # Konversi ke format yang lebih mudah diinterpretasi (0: normal, 1: anomali)
        anomaly_labels = np.where(predictions == -1, 1, 0)
        
        # Hitung anomaly score
        anomaly_scores = self.model.decision_function(data)
        # Konversi score menjadi nilai antara 0 dan 1 (semakin kecil semakin anomali)
        anomaly_scores = 1 - (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
        
        return anomaly_labels, anomaly_scores
    
    def evaluate(self, X_test, true_labels=None):
        """
        Evaluasi model pada data testing
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Data testing
        true_labels : numpy.ndarray, optional
            Label sebenarnya jika tersedia (1: anomali, 0: normal)
            
        Returns:
        --------
        dict
            Metrik evaluasi model
        """
        anomaly_labels, anomaly_scores = self.predict(X_test)
        
        # Jika true labels tersedia, hitung metrik evaluasi
        if true_labels is not None:
            cm = confusion_matrix(true_labels, anomaly_labels)
            report = classification_report(true_labels, anomaly_labels, output_dict=True)
            
            # Hitung precision-recall curve
            precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
            
            return {
                'confusion_matrix': cm,
                'classification_report': report,
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }
        
        # Jika true labels tidak tersedia, hitung statistik dasar
        anomaly_count = np.sum(anomaly_labels)
        anomaly_percentage = (anomaly_count / len(anomaly_labels)) * 100
        
        return {
            'anomaly_count': anomaly_count,
            'total_samples': len(anomaly_labels),
            'anomaly_percentage': anomaly_percentage
        }
    
    def save_model(self, model_path):
        """
        Menyimpan model ke file dalam format h5
        
        Parameters:
        -----------
        model_path : str
            Path untuk menyimpan model
        """
        try:
            # Konversi Path object ke string jika perlu
            if not isinstance(model_path, str):
                model_path = str(model_path)
                
            # Buat direktori jika belum ada
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Pastikan path berakhiran .h5
            if not model_path.endswith('.h5'):
                model_path = os.path.splitext(model_path)[0] + '.h5'
            
            # Simpan model dalam format h5
            with h5py.File(model_path, 'w') as hf:
                # Simpan parameter model
                hf.attrs['contamination'] = self.contamination
                hf.attrs['n_estimators'] = self.n_estimators
                hf.attrs['random_state'] = self.random_state
                hf.attrs['max_samples'] = str(self.max_samples)
                
                # Simpan model dengan pickle dalam dataset h5
                model_bytes = pickle.dumps(self.model)
                hf.create_dataset('model_pickle', data=np.void(model_bytes))
            
            self.model_path = model_path
            print(f"Model berhasil disimpan ke {model_path} dalam format h5")
        except Exception as e:
            print(f"Error saat menyimpan model: {e}")
            # Tampilkan informasi lebih detail untuk debugging
            import traceback
            traceback.print_exc()
    
    def load_model(self, model_path):
        """
        Memuat model dari file h5
        
        Parameters:
        -----------
        model_path : str
            Path ke file model h5
        """
        try:
            # Konversi Path object ke string jika perlu
            if not isinstance(model_path, str):
                model_path = str(model_path)
                
            # Pastikan path berakhiran .h5
            if not model_path.endswith('.h5'):
                model_path = os.path.splitext(model_path)[0] + '.h5'
                
            # Muat model dari file h5
            with h5py.File(model_path, 'r') as hf:
                # Muat parameter model
                self.contamination = hf.attrs['contamination']
                self.n_estimators = hf.attrs['n_estimators']
                self.random_state = hf.attrs['random_state']
                self.max_samples = hf.attrs['max_samples']
                
                # Muat model dari pickle dalam dataset h5
                model_bytes = hf['model_pickle'][()]
                self.model = pickle.loads(model_bytes.tobytes())
            
            self.model_path = model_path
            print(f"Model berhasil dimuat dari {model_path}")
        except Exception as e:
            print(f"Error saat memuat model: {e}")
            # Tampilkan informasi lebih detail untuk debugging
            import traceback
            traceback.print_exc()
    
    def visualize_anomalies(self, data, anomaly_labels, anomaly_scores, features=None, n_features=2):
        """
        Visualisasi anomali dalam data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data yang akan divisualisasikan
        anomaly_labels : numpy.ndarray
            Label anomali (1: anomali, 0: normal)
        anomaly_scores : numpy.ndarray
            Skor anomali
        features : list, optional
            Daftar fitur yang akan divisualisasikan
        n_features : int, default=2
            Jumlah fitur teratas berdasarkan korelasi dengan anomaly score
        """
        # Tambahkan label dan skor anomali ke data
        viz_data = data.copy()
        viz_data['anomaly'] = anomaly_labels
        viz_data['anomaly_score'] = anomaly_scores
        
        # Jika features tidak ditentukan, pilih n_features teratas berdasarkan korelasi dengan anomaly score
        if features is None:
            # Hitung korelasi antara fitur dan anomaly score
            correlations = []
            for col in data.columns:
                corr = np.abs(np.corrcoef(data[col], anomaly_scores)[0, 1])
                correlations.append((col, corr))
            
            # Urutkan fitur berdasarkan korelasi
            sorted_features = sorted(correlations, key=lambda x: x[1], reverse=True)
            features = [f[0] for f in sorted_features[:n_features]]
        
        # Buat figure dengan 3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Scatter plot untuk 2 fitur teratas
        if len(features) >= 2:
            scatter = axes[0].scatter(
                viz_data[features[0]], 
                viz_data[features[1]],
                c=viz_data['anomaly_score'],
                cmap='YlOrRd',
                alpha=0.7,
                s=50
            )
            axes[0].set_xlabel(features[0])
            axes[0].set_ylabel(features[1])
            axes[0].set_title(f'Scatter Plot Anomali ({features[0]} vs {features[1]})')
            plt.colorbar(scatter, ax=axes[0], label='Anomaly Score')
        
        # 2. Distribusi anomaly score
        sns.histplot(viz_data['anomaly_score'], kde=True, ax=axes[1])
        axes[1].axvline(x=viz_data['anomaly_score'].quantile(1-self.contamination), 
                       color='r', linestyle='--', label=f'Threshold ({self.contamination})')
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Frekuensi')
        axes[1].set_title('Distribusi Anomaly Score')
        axes[1].legend()
        
        # 3. Box plot untuk fitur berdasarkan label anomali
        if len(features) > 0:
            # Reshape data untuk seaborn boxplot
            box_data = pd.melt(viz_data[features + ['anomaly']], 
                              id_vars=['anomaly'], 
                              value_vars=features,
                              var_name='Feature', 
                              value_name='Value')
            box_data['anomaly'] = box_data['anomaly'].map({0: 'Normal', 1: 'Anomali'})
            
            sns.boxplot(x='Feature', y='Value', hue='anomaly', data=box_data, ax=axes[2])
            axes[2].set_title('Perbandingan Distribusi Fitur (Normal vs Anomali)')
            axes[2].set_xlabel('Fitur')
            axes[2].set_ylabel('Nilai')
        
        plt.tight_layout()
        return fig

# Fungsi utama untuk menjalankan deteksi anomali
def main():
    # Inisialisasi detector
    detector = AnomalyDetector(contamination=0.05, n_estimators=100)
    
    # Path ke file data
    # Gunakan Path untuk menangani path dengan benar di berbagai OS
    base_dir = Path(__file__).parent.parent.parent  # Naik 2 level dari src/models
    
    # Daftar kemungkinan lokasi file data
    possible_data_paths = [
        base_dir / "data" / "cleaned_data.csv",           # Lokasi standar: project_root/data/cleaned_data.csv
        base_dir / "cleaned_data.csv",                   # Lokasi alternatif 1: project_root/cleaned_data.csv
        base_dir / "src" / "data" / "cleaned_data.csv",  # Lokasi alternatif 2: project_root/src/data/cleaned_data.csv
        base_dir / "data" / "processed" / "cleaned_data.csv", # Lokasi alternatif 3: project_root/data/processed/cleaned_data.csv
        Path.cwd() / "cleaned_data.csv"                 # Lokasi alternatif 4: current_working_directory/cleaned_data.csv
    ]
    
    # Cari file data di semua kemungkinan lokasi
    data_path = None
    for path in possible_data_paths:
        if path.exists():
            data_path = path
            print(f"File data ditemukan di {data_path}")
            break
    
    # Jika file tidak ditemukan di semua lokasi yang dicek
    if data_path is None:
        print("File data tidak ditemukan di lokasi-lokasi berikut:")
        for path in possible_data_paths:
            print(f"- {path}")
        print("\nSilakan pastikan file cleaned_data.csv tersedia di salah satu lokasi di atas,")
        print("atau sesuaikan path di kode ini untuk mengarah ke lokasi file yang benar.")
        return
    
    # Load data
    data = detector.load_data(data_path)
    if data is None:
        return
    
    # Preprocessing data
    processed_data = detector.preprocess_data(data)
    
    # Train model
    X_train, X_test = detector.train(processed_data)
    
    # Prediksi anomali
    anomaly_labels, anomaly_scores = detector.predict(X_test)
    
    # Evaluasi model
    eval_results = detector.evaluate(X_test)
    print(f"\nHasil Deteksi Anomali:")
    print(f"Jumlah anomali terdeteksi: {eval_results['anomaly_count']} dari {eval_results['total_samples']} sampel")
    print(f"Persentase anomali: {eval_results['anomaly_percentage']:.2f}%")
    
    # Visualisasi hasil
    fig = detector.visualize_anomalies(X_test, anomaly_labels, anomaly_scores)
    
    # Simpan model dalam format h5 di folder data/Model-output
    model_dir = base_dir / "data" / "Model-output"
    # Buat direktori jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    model_path = str(model_dir / "isolation_forest_model.h5")
    print(f"Menyimpan model ke: {model_path}")
    detector.save_model(model_path)
    
    # Tampilkan visualisasi
    plt.show()

# Jalankan program jika dieksekusi langsung
if __name__ == "__main__":
    main()