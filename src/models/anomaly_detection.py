import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

class AviationAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.scaler = StandardScaler()
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        
    def preprocess_data(self, data):
        """Preprocess flight data by scaling numerical features"""
        return self.scaler.fit_transform(data)
    
    def detect_anomalies(self, flight_data):
        """Detect anomalies in flight data"""
        # Preprocess the data
        scaled_data = self.preprocess_data(flight_data)
        
        # Fit and predict anomalies
        predictions = self.detector.fit_predict(scaled_data)
        
        # Convert predictions to binary (1 for normal, 0 for anomaly)
        # IsolationForest returns 1 for normal and -1 for anomaly
        anomalies = np.where(predictions == -1, 1, 0)
        
        return anomalies
    
    def analyze_results(self, true_labels, predicted_anomalies):
        """Analyze detection results"""
        return classification_report(true_labels, predicted_anomalies)

# Example usage
if __name__ == "__main__":
    # Sample data (replace with real flight data)
    sample_data = pd.DataFrame({
        'altitude': np.random.normal(30000, 1000, 1000),
        'speed': np.random.normal(500, 50, 1000),
        'vertical_speed': np.random.normal(0, 100, 1000),
        'heading': np.random.normal(180, 30, 1000)
    })
    
    # Create and use detector
    detector = AviationAnomalyDetector()
    anomalies = detector.detect_anomalies(sample_data)
    
    # Print results
    print(f"Number of anomalies detected: {sum(anomalies)}")
    print(f"Percentage of flights flagged as anomalous: {(sum(anomalies)/len(anomalies))*100:.2f}%")
