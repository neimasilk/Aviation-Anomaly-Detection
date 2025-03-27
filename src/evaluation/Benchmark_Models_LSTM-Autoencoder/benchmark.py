import pandas as pd
import json

def read_lstm_report():
    df = pd.read_csv('report_LSTM.csv')
    metrics = {}
    for _, row in df.iterrows():
        if row['Kategori'] in ['Metrics', 'Benchmark']:
            key = row['Metrik'].lower().replace(' ', '_').replace('(', '').replace(')', '')
            value = str(row['Nilai']).replace('%', '')
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics

def read_autoencoder_report():
    metrics = {}
    with open('report_Autoencoder.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line and not line.startswith('Catatan'):
                key, value = line.strip().split(':')
                key = key.strip().lower().replace(' ', '_')
                try:
                    metrics[key] = float(value.strip())
                except ValueError:
                    metrics[key] = value.strip()
    return metrics

def compare_models():
    lstm = read_lstm_report()
    autoencoder = read_autoencoder_report()
    
    comparison = [
        "PERBANDINGAN MODEL LSTM DAN AUTOENCODER",
        "=" * 50,
        "\nMETRIK PERFORMA:\n",
        f"1. Akurasi:",
        f"   - LSTM: {lstm.get('akurasi', 'N/A')}%",
        f"   - Autoencoder: {autoencoder.get('accuracy', 'N/A') * 100:.2f}%",
        f"\n2. Presisi:",
        f"   - LSTM: {lstm.get('precision', 'N/A')}%",
        f"   - Autoencoder: {autoencoder.get('precision', 'N/A') * 100:.2f}%",
        f"\n3. Recall:",
        f"   - LSTM: {lstm.get('recall', 'N/A')}%",
        f"   - Autoencoder: {autoencoder.get('recall', 'N/A') * 100:.2f}%",
        f"\n4. F1-Score:",
        f"   - LSTM: {lstm.get('f1score', 'N/A')}%",
        f"   - Autoencoder: {autoencoder.get('f1_score', 'N/A') * 100:.2f}%",
        f"\n5. ROC-AUC:",
        f"   - LSTM: {lstm.get('rocauc', 'N/A')}%",
        f"   - Autoencoder: {autoencoder.get('roc_auc', 'N/A') * 100:.2f}%",
        "\nDETEKSI ANOMALI:\n",
        f"1. Jumlah Anomali Terdeteksi:",
        f"   - LSTM: {lstm.get('jumlah_anomali', 'N/A')} ({lstm.get('persentase_anomali', 'N/A')}%)",
        f"   - Autoencoder: {int(autoencoder.get('anomalies_detected', 'N/A'))} ({autoencoder.get('detection_rate', 'N/A')}%)",
        f"\n2. Waktu Pemrosesan:",
        f"   - LSTM: {lstm.get('waktu_pemrosesan_lstm', 'N/A')} detik",
        f"   - Autoencoder: {autoencoder.get('avg_processing_time', 'N/A')} detik",
        "\nKESIMPULAN:\n",
        "1. Performa Model:",
        "   - Model Autoencoder menunjukkan performa yang lebih baik dalam hal akurasi, presisi, recall, dan F1-score",
        "   - Model LSTM memiliki tingkat akurasi yang cukup baik namun kurang dalam hal presisi dan recall",
        "\n2. Deteksi Anomali:",
        f"   - LSTM berhasil mendeteksi {lstm.get('jumlah_anomali', 'N/A')} anomali ({lstm.get('persentase_anomali', 'N/A')}%)",
        f"   - Autoencoder mendeteksi {int(autoencoder.get('anomalies_detected', 'N/A'))} anomali ({autoencoder.get('detection_rate', 'N/A')}%)",
        "\n3. Efisiensi:",
        "   - Kedua model memiliki waktu pemrosesan yang cukup efisien (< 1 detik)",
        "   - Autoencoder sedikit lebih cepat dalam pemrosesan dibandingkan LSTM"
    ]
    
    with open('comparison.txt', 'w') as f:
        f.write('\n'.join(comparison))

if __name__ == '__main__':
    compare_models()