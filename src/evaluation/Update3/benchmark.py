import time
import argparse
import numpy as np
from metrics import load_model_from_file, load_and_preprocess_data, calculate_metrics

def analyze_model_performance(model_path, data_path):
    """Analyze model performance including detection rate, anomaly count, and processing time.
    
    Args:
        model_path (str): Path to the trained model file
        data_path (str): Path to the test data file
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Load model and data
    model = load_model_from_file(model_path)
    X, y_true = load_and_preprocess_data(data_path)
    
    # Initialize metrics
    total_samples = len(X)
    processing_times = []
    
    # Process each sample and measure time
    predictions = []
    for i in range(total_samples):
        sample = X[i:i+1]  # Get single sample
        
        # Measure processing time
        start_time = time.time()
        pred = model.predict(sample, verbose=0)
        end_time = time.time()
        
        predictions.append(pred[0])
        processing_times.append(end_time - start_time)
    
    # Convert predictions to numpy array
    y_pred = np.array(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Calculate average processing time
    avg_processing_time = np.mean(processing_times)
    
    # Count detected anomalies
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred
    
    anomalies_detected = np.sum(y_pred_classes == 1)  # Assuming 1 represents anomaly
    
    # Get detection rate (recall)
    detection_rate = metrics.get('recall', 0) * 100
    
    return {
        'detection_rate': detection_rate,
        'anomalies_detected': int(anomalies_detected),
        'avg_processing_time': avg_processing_time
    }

def generate_report(metrics):
    """Generate a human-readable report from the metrics.
    
    Args:
        metrics (dict): Dictionary containing performance metrics
        
    Returns:
        str: Formatted report string
    """
    return (
        f"Laporan ini menyimpulkan bahwa model bisa mendeteksi {metrics['detection_rate']:.1f}% anomali "
        f"dan sebanyak {metrics['anomalies_detected']} Anomali terdeteksi juga dengan "
        f"rata-rata {metrics['avg_processing_time']:.1f} detik per komunikasi."
    )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze model performance and generate report')
    parser.add_argument('--model', required=True, help='Path to the .h5 model file')
    parser.add_argument('--data', required=True, help='Path to the test data file')
    
    args = parser.parse_args()
    
    try:
        # Analyze model performance
        performance_metrics = analyze_model_performance(args.model, args.data)
        
        # Generate and print report
        report = generate_report(performance_metrics)
        print("\nLaporan Kinerja Model:")
        print("-" * 50)
        print(report)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()