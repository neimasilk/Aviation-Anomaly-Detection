import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metrics import load_model_from_file, load_and_preprocess_data

# Enable eager execution
tf.config.run_functions_eagerly(True)

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation using KFold.
    
    Args:
        model: TensorFlow/Keras model
        X: Input features
        y: Target labels
        cv (int): Number of folds (default: 5)
        
    Returns:
        dict: Dictionary containing cross-validation metrics
    """
    try:
        # Convert inputs to tensors
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            # Split data
            X_train = tf.gather(X, train_idx)
            X_val = tf.gather(X, val_idx)
            y_train = tf.gather(y, train_idx)
            y_val = tf.gather(y, val_idx)
            
            # Recompile model for each fold
            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            
            # Train model on fold
            model.fit(X_train, y_train, verbose=0)
            
            # Get predictions
            y_pred = model.predict(X_val, verbose=0)
            
            # Convert tensors to numpy for metric calculation
            y_val_np = y_val.numpy()
            y_pred_np = y_pred
            
            # Calculate metrics for fold
            metrics = calculate_metrics(y_val_np, y_pred_np)
            metrics['fold'] = fold
            fold_metrics.append(metrics)
        
        # Calculate average metrics across folds
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in fold_metrics])
        }
        
        return {
            'fold_metrics': fold_metrics,
            'average_metrics': avg_metrics
        }
        
    except Exception as e:
        raise Exception(f"Error during cross-validation: {str(e)}")

def test_model_on_data(model, X_test, y_test):
    """Test model on independent test data.
    
    Args:
        model: TensorFlow/Keras model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing test metrics
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Add latency measurement
        latency = measure_latency(model, X_test[:1])  # Test latency on single sample
        metrics['average_latency'] = latency
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Error during model testing: {str(e)}")

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        
    Returns:
        dict: Dictionary containing various metrics
    """
    try:
        # Convert probabilities to class labels if needed
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Error calculating metrics: {str(e)}")

def measure_latency(model, X_sample, n_iterations=100):
    """Measure model processing time.
    
    Args:
        model: TensorFlow/Keras model
        X_sample: Single sample for prediction
        n_iterations (int): Number of iterations for averaging
        
    Returns:
        float: Average processing time in seconds
    """
    try:
        processing_times = []
        
        # Warm-up prediction
        model.predict(X_sample, verbose=0)
        
        # Measure processing time over multiple iterations
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X_sample, verbose=0)
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Calculate average processing time
        avg_processing_time = np.mean(processing_times)
        
        return float(avg_processing_time)
        
    except Exception as e:
        raise Exception(f"Error measuring latency: {str(e)}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Validate model using cross-validation and testing')
    parser.add_argument('--model', required=True, help='Path to the .h5 model file')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    try:
        # Load model and data
        model = load_model_from_file(args.model)
        X, y = load_and_preprocess_data(args.data)
        
        # Perform cross-validation
        print("\nMenjalankan Validasi Silang...")
        cv_results = cross_validate_model(model, X, y, cv=args.cv)
        
        # Print cross-validation results
        print("\nHasil Validasi Silang:")
        print("-" * 50)
        for fold_metric in cv_results['fold_metrics']:
            print(f"Fold {fold_metric['fold']}:")
            print(f"  Akurasi: {fold_metric['accuracy']:.4f}")
            print(f"  Presisi: {fold_metric['precision']:.4f}")
            print(f"  Recall: {fold_metric['recall']:.4f}")
            print(f"  F1-score: {fold_metric['f1_score']:.4f}")
        
        print("\nRata-rata Metrik:")
        print("-" * 50)
        avg = cv_results['average_metrics']
        print(f"Akurasi: {avg['accuracy']:.4f}")
        print(f"Presisi: {avg['precision']:.4f}")
        print(f"Recall: {avg['recall']:.4f}")
        print(f"F1-score: {avg['f1_score']:.4f}")
        
        # Test model on full dataset
        print("\nMenjalankan Pengujian Model...")
        test_results = test_model_on_data(model, X, y)
        
        print("\nHasil Pengujian:")
        print("-" * 50)
        print(f"Akurasi: {test_results['accuracy']:.4f}")
        print(f"Presisi: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"F1-score: {test_results['f1_score']:.4f}")
        print(f"Rata-rata Latensi: {test_results['average_latency']:.4f} detik")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()