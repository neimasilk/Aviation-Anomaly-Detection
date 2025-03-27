import pandas as pd
import numpy as np

def load_metrics():
    """Load data dari hasil_metrics.csv"""
    try:
        df = pd.read_csv('hasil_metrics.csv')
        if 'Metrik' not in df.columns or 'Nilai' not in df.columns:
            print('Format file hasil_metrics.csv tidak sesuai')
            return None
        return df
    except FileNotFoundError:
        print('File hasil_metrics.csv tidak ditemukan')
        return None

def load_validation():
    """Load data dari hasil_validasi.csv"""
    try:
        df = pd.read_csv('hasil_validasi.csv')
        if 'Metrik' not in df.columns or 'Nilai' not in df.columns:
            print('Format file hasil_validasi.csv tidak sesuai')
            return None
        return df
    except FileNotFoundError:
        print('File hasil_validasi.csv tidak ditemukan')
        return None

def load_benchmark():
    """Load data dari benchmark_results.csv"""
    try:
        df = pd.read_csv('benchmark_results.csv')
        if 'Metrik' not in df.columns or 'Nilai' not in df.columns:
            print('Format file benchmark_results.csv tidak sesuai')
            return None
        return df
    except FileNotFoundError:
        print('File benchmark_results.csv tidak ditemukan')
        return None

def generate_report():
    # Load semua data
    metrics_df = load_metrics()
    validation_df = load_validation()
    benchmark_df = load_benchmark()
    
    # Inisialisasi list untuk menyimpan hasil report
    report_data = {
        'Kategori': [],
        'Metrik': [],
        'Nilai': []
    }
    
    # Tambahkan data metrics
    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            report_data['Kategori'].append('Metrics')
            report_data['Metrik'].append(row['Metrik'])
            report_data['Nilai'].append(row['Nilai'])
    
    # Tambahkan data validasi
    if validation_df is not None:
        for _, row in validation_df.iterrows():
            report_data['Kategori'].append('Validasi')
            report_data['Metrik'].append(row['Metrik'])
            report_data['Nilai'].append(row['Nilai'])
    
    # Tambahkan data benchmark
    if benchmark_df is not None:
        for _, row in benchmark_df.iterrows():
            report_data['Kategori'].append('Benchmark')
            report_data['Metrik'].append(row['Metrik'])
            report_data['Nilai'].append(row['Nilai'])
    
    # Buat DataFrame dari hasil report
    report_df = pd.DataFrame(report_data)
    
    # Simpan report ke CSV
    report_df.to_csv('report_result.csv', index=False)
    
    print('\nLaporan telah berhasil dibuat dan disimpan dalam file report_result.csv')
    print('\nRingkasan Laporan:')
    print('=' * 50)
    
    # Tampilkan ringkasan berdasarkan kategori
    for kategori in report_df['Kategori'].unique():
        print(f'\n{kategori}:')
        kategori_data = report_df[report_df['Kategori'] == kategori]
        for _, row in kategori_data.iterrows():
            print(f"{row['Metrik']}: {row['Nilai']}")

if __name__ == '__main__':
    generate_report()