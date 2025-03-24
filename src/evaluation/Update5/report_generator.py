import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_metrics_data(csv_path):
    """Membaca data metrik dari file CSV.
    
    Args:
        csv_path (str): Path ke file CSV yang berisi metrik evaluasi
        
    Returns:
        pd.DataFrame: DataFrame berisi metrik dan nilainya
    """
    try:
        metrics_df = pd.read_csv(csv_path)
        print(f"Data metrik berhasil dibaca dari {csv_path}")
        return metrics_df
    except Exception as e:
        raise Exception(f"Error membaca file CSV: {str(e)}")

def generate_text_report(metrics_df, output_path, benchmark_metrics=None):
    """Menghasilkan laporan teks berisi ringkasan metrik.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame berisi metrik evaluasi
        output_path (str): Path untuk menyimpan file laporan teks
        benchmark_metrics (dict, optional): Metrik tambahan dari benchmark
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LAPORAN EVALUASI MODEL\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Ringkasan Metrik:\n")
            f.write("-" * 20 + "\n")
            
            # Tulis setiap metrik dan nilainya
            for _, row in metrics_df.iterrows():
                metric_name = row['metric']
                metric_value = row['value']
                f.write(f"{metric_name}: {metric_value:.4f}\n")
            
            # Tambahkan informasi waktu proses dan statistik anomali
            if benchmark_metrics:
                f.write("\nWaktu Proses (Latency):\n")
                f.write(f"Rata-rata waktu per komunikasi: {benchmark_metrics['avg_processing_time']:.1f} detik\n")
                f.write(f"Memenuhi syarat <1 detik: {'Ya' if benchmark_metrics['avg_processing_time'] < 1 else 'Tidak'}\n")
                
                f.write("\nJumlah Anomali Terdeteksi:\n")
                f.write(f"Total data uji: {benchmark_metrics['total_samples']} komunikasi\n")
                f.write(f"Jumlah anomali nyata: {benchmark_metrics['true_anomalies']}\n")
                f.write(f"Jumlah terdeteksi: {benchmark_metrics['anomalies_detected']}\n")
            
            f.write("\nCatatan:\n")
            f.write("- Nilai metrik di atas menunjukkan performa model pada data evaluasi\n")
            f.write("- Semakin tinggi nilai metrik, semakin baik performa model\n")
        
        print(f"Laporan teks berhasil disimpan ke {output_path}")
    except Exception as e:
        raise Exception(f"Error membuat laporan teks: {str(e)}")

def create_metrics_visualization(metrics_df, output_path):
    """Membuat visualisasi grafik batang untuk metrik evaluasi.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame berisi metrik evaluasi
        output_path (str): Path untuk menyimpan file visualisasi
    """
    try:
        # Set style seaborn
        sns.set_style("whitegrid")
        
        # Buat figure dengan ukuran yang sesuai
        plt.figure(figsize=(10, 6))
        
        # Buat bar plot
        bar_plot = sns.barplot(
            x='metric',
            y='value',
            hue='metric',
            data=metrics_df,
            palette='viridis',
            legend=False
        )
        
        # Kustomisasi plot
        plt.title('Ringkasan Metrik Evaluasi Model', pad=20)
        plt.xlabel('Metrik')
        plt.ylabel('Nilai')
        
        # Rotasi label sumbu x untuk keterbacaan lebih baik
        plt.xticks(rotation=45)
        
        # Tambahkan nilai di atas setiap bar
        for i, v in enumerate(metrics_df['value']):
            bar_plot.text(
                i, v, 
                f'{v:.4f}', 
                ha='center',
                va='bottom'
            )
        
        # Sesuaikan layout dan simpan plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisasi metrik berhasil disimpan ke {output_path}")
    except Exception as e:
        raise Exception(f"Error membuat visualisasi: {str(e)}")

def main():
    try:
        # Baca data metrik
        metrics_df = read_metrics_data('metrics.csv')
        
        # Generate laporan teks
        generate_text_report(metrics_df, 'evaluation_report.txt')
        
        # Buat visualisasi
        create_metrics_visualization(metrics_df, 'metrics_summary.png')
        
        print("\nPembuatan laporan evaluasi selesai!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()