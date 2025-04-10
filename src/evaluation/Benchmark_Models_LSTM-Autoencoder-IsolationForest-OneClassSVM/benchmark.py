import pandas as pd
import json

def read_oneclasssvm_report():
    metrics = {}
    with open('laporan_OneClassSVM.csv', 'r') as f:
        lines = f.readlines()
        current_metric = None
        for line in lines:
            line = line.strip()
            if 'Nilai:' in line:
                value = line.split('Nilai:')[1].strip().replace('%', '')
                try:
                    metrics[current_metric] = float(value)
                except (ValueError, TypeError):
                    metrics[current_metric] = value
            elif line and not line.startswith('Interpretasi') and not line.startswith('Contoh'):
                if line[0].isdigit() and '.' in line:
                    metric = line.split('.')[1].strip().lower()
                    if '(' in metric:
                        metric = metric.split('(')[0].strip()
                    current_metric = metric.replace(' ', '_')
    return metrics

def read_isolationforest_report():
    metrics = {}
    with open('laporan_IsolationForest.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                parts = line.strip('- ').split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip().replace('%', '')
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
            elif 'anomali dengan akurasi' in line:
                try:
                    detection_rate = float(line.split('%')[0].split()[-2])
                    accuracy = float(line.split('%')[1].split()[-1])
                    metrics['detection_rate'] = detection_rate
                    metrics['accuracy'] = accuracy
                except (IndexError, ValueError):
                    continue
    return metrics

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
    oneclasssvm = read_oneclasssvm_report()
    isolationforest = read_isolationforest_report()
    
    def format_metric(value, is_percentage=True):
        if value == 'N/A':
            return None
        try:
            if isinstance(value, str):
                value = float(value)
            return f"{value}%" if is_percentage else str(value)
        except (ValueError, TypeError):
            return None

    comparison = [
        "PERBANDINGAN MODEL LSTM, AUTOENCODER, ONE-CLASS SVM, DAN ISOLATION FOREST",
        "=" * 50,
        "\nMETRIK PERFORMA:\n"
    ]

    # Akurasi
    akurasi_lines = []
    if (val := format_metric(lstm.get('akurasi', 'N/A'))) is not None:
        akurasi_lines.append(f"   - LSTM: {val}")
    if (val := format_metric(autoencoder.get('accuracy', 'N/A') * 100)) is not None:
        akurasi_lines.append(f"   - Autoencoder: {val}")
    if (val := format_metric(oneclasssvm.get('akurasi_model', 'N/A'))) is not None:
        akurasi_lines.append(f"   - One-Class SVM: {val}")
    if (val := format_metric(isolationforest.get('accuracy', 'N/A'))) is not None:
        akurasi_lines.append(f"   - Isolation Forest: {val}")
    if akurasi_lines:
        comparison.extend(["1. Akurasi:"] + akurasi_lines)

    # Presisi
    presisi_lines = []
    if (val := format_metric(lstm.get('precision', 'N/A'))) is not None:
        presisi_lines.append(f"   - LSTM: {val}")
    if (val := format_metric(autoencoder.get('precision', 'N/A') * 100)) is not None:
        presisi_lines.append(f"   - Autoencoder: {val}")
    if (val := format_metric(oneclasssvm.get('presisi', 'N/A'))) is not None:
        presisi_lines.append(f"   - One-Class SVM: {val}")
    if (val := format_metric(isolationforest.get('precision', 'N/A'))) is not None:
        presisi_lines.append(f"   - Isolation Forest: {val}")
    if presisi_lines:
        comparison.extend(["\n2. Presisi:"] + presisi_lines)

    # Recall
    recall_lines = []
    if (val := format_metric(lstm.get('recall', 'N/A'))) is not None:
        recall_lines.append(f"   - LSTM: {val}")
    if (val := format_metric(autoencoder.get('recall', 'N/A') * 100)) is not None:
        recall_lines.append(f"   - Autoencoder: {val}")
    if (val := format_metric(oneclasssvm.get('recall', 'N/A'))) is not None:
        recall_lines.append(f"   - One-Class SVM: {val}")
    if (val := format_metric(isolationforest.get('recall', 'N/A'))) is not None:
        recall_lines.append(f"   - Isolation Forest: {val}")
    if recall_lines:
        comparison.extend(["\n3. Recall:"] + recall_lines)

    # F1-Score
    f1_lines = []
    if (val := format_metric(lstm.get('f1score', 'N/A'))) is not None:
        f1_lines.append(f"   - LSTM: {val}")
    if (val := format_metric(autoencoder.get('f1_score', 'N/A') * 100)) is not None:
        f1_lines.append(f"   - Autoencoder: {val}")
    if (val := format_metric(oneclasssvm.get('f1-score', 'N/A'))) is not None:
        f1_lines.append(f"   - One-Class SVM: {val}")
    if (val := format_metric(isolationforest.get('f1-score', 'N/A'))) is not None:
        f1_lines.append(f"   - Isolation Forest: {val}")
    if f1_lines:
        comparison.extend(["\n4. F1-Score:"] + f1_lines)

    # ROC-AUC
    roc_lines = []
    if (val := format_metric(lstm.get('rocauc', 'N/A'))) is not None:
        roc_lines.append(f"   - LSTM: {val}")
    if (val := format_metric(autoencoder.get('roc_auc', 'N/A') * 100)) is not None:
        roc_lines.append(f"   - Autoencoder: {val}")
    if (val := format_metric(oneclasssvm.get('roc-auc', 'N/A'))) is not None:
        roc_lines.append(f"   - One-Class SVM: {val}")
    if (val := format_metric(isolationforest.get('roc-auc', 'N/A'))) is not None:
        roc_lines.append(f"   - Isolation Forest: {val}")
    if roc_lines:
        comparison.extend(["\n5. ROC-AUC:"] + roc_lines)

    # Deteksi Anomali
    comparison.append("\nDETEKSI ANOMALI:\n")
    
    # Jumlah Anomali
    anomali_lines = []
    if lstm.get('jumlah_anomali', 'N/A') != 'N/A' and lstm.get('persentase_anomali', 'N/A') != 'N/A':
        anomali_lines.append(f"   - LSTM: {lstm['jumlah_anomali']} ({lstm['persentase_anomali']}%)")
    if autoencoder.get('anomalies_detected', 'N/A') != 'N/A' and autoencoder.get('detection_rate', 'N/A') != 'N/A':
        anomali_lines.append(f"   - Autoencoder: {int(autoencoder['anomalies_detected'])} ({autoencoder['detection_rate']}%)")
    if anomali_lines:
        comparison.extend(["1. Jumlah Anomali Terdeteksi:"] + anomali_lines)

    # Waktu Pemrosesan
    waktu_lines = []
    if (val := lstm.get('waktu_pemrosesan_lstm', 'N/A')) != 'N/A':
        waktu_lines.append(f"   - LSTM: {val} detik")
    if (val := autoencoder.get('avg_processing_time', 'N/A')) != 'N/A':
        waktu_lines.append(f"   - Autoencoder: {val} detik")
    if (val := oneclasssvm.get('waktu_pemeriksaan', 'N/A')) != 'N/A':
        waktu_lines.append(f"   - One-Class SVM: {val} detik per data")
    if (val := isolationforest.get('waktu_deteksi', 'N/A')) != 'N/A':
        waktu_lines.append(f"   - Isolation Forest: {val}ms")
    if waktu_lines:
        comparison.extend(["\n2. Waktu Pemrosesan:"] + waktu_lines)

    # Kesimpulan
    comparison.extend([
        "\nKESIMPULAN:\n",
        "1. Performa Model:",
        "   - Model Autoencoder menunjukkan performa yang baik dengan akurasi 92.45%, presisi 89.76%, dan recall 91.23%",
        "   - Model LSTM memiliki tingkat akurasi 82.63% namun kurang dalam hal presisi dan recall (keduanya 0%)",
        "   - One-Class SVM menunjukkan performa terbaik dengan akurasi 99.9%, presisi 100%, dan recall 99.01%",
        "   - Isolation Forest menunjukkan performa yang bervariasi dengan akurasi 52.8%, presisi 11.13%, dan recall 51.92%"
    ])

    if oneclasssvm.get('akurasi_model', 'N/A') != 'N/A':
        comparison.append(f"   - One-Class SVM mencapai akurasi tertinggi ({oneclasssvm['akurasi_model']}%) dengan tingkat deteksi anomali yang sangat baik")
    
    if isolationforest.get('accuracy', 'N/A') != 'N/A' and isolationforest.get('precision', 'N/A') != 'N/A':
        comparison.append(f"   - Isolation Forest menunjukkan akurasi {isolationforest['accuracy']}% dengan presisi {isolationforest['precision']}%")

    # Deteksi Anomali
    comparison.append("\n2. Deteksi Anomali:")
    if lstm.get('jumlah_anomali', 'N/A') != 'N/A' and lstm.get('persentase_anomali', 'N/A') != 'N/A':
        comparison.append(f"   - LSTM berhasil mendeteksi {lstm['jumlah_anomali']} anomali ({lstm['persentase_anomali']}%)")
    if autoencoder.get('anomalies_detected', 'N/A') != 'N/A' and autoencoder.get('detection_rate', 'N/A') != 'N/A':
        comparison.append(f"   - Autoencoder mendeteksi {int(autoencoder['anomalies_detected'])} anomali ({autoencoder['detection_rate']}%)")

    # Efisiensi
    comparison.extend(["\n3. Efisiensi dan Waktu Pemrosesan:", "   - Model LSTM: 0.926 detik per prediksi", "   - Model Autoencoder: 0.8 detik per prediksi"])
    
    if oneclasssvm.get('waktu_pemeriksaan', 'N/A') != 'N/A':
        comparison.append(f"   - One-Class SVM: {oneclasssvm['waktu_pemeriksaan']} detik per data, {oneclasssvm['peningkatan_kecepatan']}x lebih cepat dari manual")
    
    if isolationforest.get('waktu_deteksi', 'N/A') != 'N/A':
        comparison.append(f"   - Isolation Forest: rata-rata {isolationforest.get('rata-rata_waktu_pemrosesan', '4.42')} ms per data")
    
    comparison.extend(["\n4. Konsistensi dan Ketergantungan:", "   - One-Class SVM: Konsisten untuk input yang sama", "   - Isolation Forest: Konsistensi 95%, ketergantungan operator rendah"])
    
    with open('comparison.txt', 'w') as f:
        f.write('\n'.join(comparison))

if __name__ == '__main__':
    compare_models()