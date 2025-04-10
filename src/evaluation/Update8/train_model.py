import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from joblib import dump

# Baca data
df = pd.read_csv('cleaned_data.csv')

# Pilih fitur yang akan digunakan (menggunakan kolom numerik yang tersedia)
features = ['START_TIME', 'END_TIME']
X = df[features]

# Preprocessing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inisialisasi dan latih model One Class SVM
model = OneClassSVM(
    kernel='rbf',  # Radial Basis Function kernel
    nu=0.1,        # Proporsi outlier yang diharapkan
    gamma='scale'   # Parameter kernel
)

# Latih model
model.fit(X_scaled)

# Simpan model dan scaler
dump(model, 'one_class_svm.h5')
dump(scaler, 'scaler.h5')

print('Model berhasil dilatih dan disimpan sebagai one_class_svm.h5')
print('Scaler disimpan sebagai scaler.h5')