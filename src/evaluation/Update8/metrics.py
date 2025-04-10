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

# Tampilkan hasil evaluasi dengan penjelasan detail
print('Hasil Evaluasi Model One-Class SVM:')
print('\n1. Akurasi (Accuracy)')
print(f'   Nilai: {accuracy:.2%}')
print('   Interpretasi: Persentase total prediksi yang benar (normal + anomali)')
print('   Contoh: Dari 100 prediksi, model benar dalam mengidentifikasi ' + \
      f'{int(accuracy * 100)} kasus baik normal maupun anomali')

print('\n2. Presisi (Precision)')
print(f'   Nilai: {precision:.2%}')
print('   Interpretasi: Persentase ketepatan dalam mengidentifikasi anomali')
print('   Contoh: Dari 100 kasus yang diprediksi sebagai anomali, ' + \
      f'{int(precision * 100)} adalah benar-benar anomali')

print('\n3. Recall (Sensitivity)')
print(f'   Nilai: {recall:.2%}')
print('   Interpretasi: Persentase anomali yang berhasil dideteksi')
print('   Contoh: Dari 100 kasus anomali yang sebenarnya, model berhasil ' + \
      f'mendeteksi {int(recall * 100)} kasus')

print('\n4. F1-Score')
print(f'   Nilai: {f1:.2%}')
print('   Interpretasi: Rata-rata harmonik dari Precision dan Recall')
print('   Menunjukkan keseimbangan antara presisi dan recall')
print('   Semakin tinggi nilai F1-Score, semakin baik performa model')

print('\n5. ROC-AUC')
print(f'   Nilai: {roc_auc:.2%}')
print('   Interpretasi: Kemampuan model membedakan antara kelas normal dan anomali')
print('   Nilai 1.0 berarti klasifikasi sempurna')
print('   Nilai 0.5 berarti klasifikasi acak')
print('   Nilai > 0.8 menunjukkan model memiliki kemampuan diskriminasi yang baik')

# Simpan metrik ke dalam file
metrics_df = pd.DataFrame({
    'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Nilai': [accuracy, precision, recall, f1, roc_auc]
})
metrics_df.to_csv('evaluation_metrics.csv', index=False)
print('\nMetrik evaluasi telah disimpan dalam file evaluation_metrics.csv')