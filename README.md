# Real-Time Anomaly Detection in Aviation Communications
Proyek ini bertujuan mendeteksi anomali secara real-time dalam komunikasi penerbangan menggunakan machine learning.

## Teknologi
- Python
- TensorFlow
- Pandas
- Scikit-learn

## Cara Menjalankan
1. Clone repositori:
   ```bash
   git clone https://github.com/[username]/Aviation-Anomaly-Detection.git
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan program:
   ```bash
   python src/main.py
   ```

## Struktur Folder
- `data/`: Dataset mentah dan yang diproses
  - `raw/`: Data mentah sebelum preprocessing
  - `processed/`: Data yang telah diproses untuk model
- `src/`: Kode sumber untuk preprocessing, model, dan evaluasi
  - `preprocessing/`: Modul untuk preprocessing data
  - `models/`: Implementasi model machine learning
  - `evaluation/`: Kode untuk evaluasi performa model
  - `utils/`: Fungsi utilitas yang digunakan di berbagai modul
- `notebooks/`: Eksperimen awal dan analisis eksploratori
- `docs/`: Dokumentasi tambahan
- `tests/`: Unit test dan integration test

## Tim
- Amien: Pengelola proyek dan dokumentasi
- Yesinka: Preprocessing data
- Yosafat: Pengembangan model
- Samuel: Evaluasi model