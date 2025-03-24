import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_model_from_file(model_path):
    """Load a trained model from an H5 file.
    
    Args:
        model_path (str): Path to the .h5 model file
        
    Returns:
        model: Loaded Keras model
    """
    try:
        # Import necessary metrics
        from tensorflow.keras.metrics import MeanSquaredError
        from tensorflow.keras.losses import mse
        
        # Custom objects dictionary for model loading
        custom_objects = {
            'mse': mse,
            'mean_squared_error': MeanSquaredError()
        }
        
        # Load model with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Detailed error information:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise Exception(f"Error loading model: {str(e)}")

def load_and_preprocess_data(data_path):
    """Load and preprocess data from a CSV file.
    
    Args:
        data_path (str): Path to the CSV data file
        
    Returns:
        tuple: Preprocessed features and labels
    """
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # For autoencoder, we'll use 'cleaned_text' as both input and output
        if 'cleaned_text' in data.columns:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Initialize and fit tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(data['cleaned_text'])
            
            # Convert texts to sequences and pad them
            sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
            padded_sequences = pad_sequences(sequences, maxlen=30, padding='post', truncating='post')
            
            # Convert to float32 for model compatibility
            padded_sequences = padded_sequences.astype('float32')
            
            # Return same data for input and target (autoencoder)
            return padded_sequences, padded_sequences
        else:
            # Fallback to original behavior for other types of models
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            return X, y
            
    except Exception as e:
        print(f"Error details:")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Data types: {data.dtypes}")
        raise Exception(f"Error loading or preprocessing data: {str(e)}")

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        
    Returns:
        dict: Dictionary containing various metrics
    """
    try:
        metrics = {}
        
        # Calculate classification metrics if input is 2D array for reconstruction
        if isinstance(y_true, np.ndarray) and y_true.ndim == 2 and y_true.shape == y_pred.shape:
            # Calculate reconstruction error and threshold for classification
            reconstruction_error = np.mean(np.square(y_true - y_pred), axis=1)
            threshold = np.percentile(reconstruction_error, 95)  # Using 95th percentile as threshold
            y_true_binary = np.zeros_like(reconstruction_error)
            y_pred_binary = (reconstruction_error > threshold).astype(int)
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            
            metrics.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
        
        # Handle classification metrics
        try:
            # Get predicted probabilities for ROC-AUC if available
            y_pred_proba = y_pred if y_pred.ndim > 1 and y_pred.shape[1] > 1 else None
            
            # Convert probabilities to class labels if needed
            y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 and y_pred.shape[1] > 1 else y_pred
            y_true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 and y_true.shape[1] > 1 else y_true
            
            # Calculate basic classification metrics
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
            
            metrics.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
            
            # Calculate ROC-AUC if we have probability predictions
            if y_pred_proba is not None:
                # For binary classification
                if y_pred_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true_classes, y_pred_proba[:, 1])
                    metrics['roc_auc'] = float(roc_auc)
                # For multi-class
                else:
                    roc_auc = roc_auc_score(y_true_classes, y_pred_proba, multi_class='ovr', average='weighted')
                    metrics['roc_auc'] = float(roc_auc)
            
            return metrics
            
        except Exception as e:
            print(f"Note: Could not calculate classification metrics: {str(e)}")
            return metrics
        
    except Exception as e:
        raise Exception(f"Error calculating metrics: {str(e)}")
    # This except clause should be removed since the exception is already handled in the try block above
        raise Exception(f"Error calculating metrics: {str(e)}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str, optional): Path to save the confusion matrix plot
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
    except Exception as e:
        raise Exception(f"Error plotting confusion matrix: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate metrics for a machine learning model')
    parser.add_argument('--model', required=True, help='Path to the .h5 model file')
    parser.add_argument('--data', required=True, help='Path to the CSV data file')
    parser.add_argument('--save-cm', help='Path to save confusion matrix plot')
    
    args = parser.parse_args()
    
    try:
        # Load model and data
        model = load_model_from_file(args.model)
        X, y_true = load_and_preprocess_data(args.data)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print("-" * 50)
        
        # Group metrics by type
        autoencoder_metrics = ['mse', 'mae', 'rmse']
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Print autoencoder metrics if available
        autoencoder_available = any(metric in metrics for metric in autoencoder_metrics)
        if autoencoder_available:
            print("\nAutoencoder Metrics:")
            print("-" * 20)
            for metric in autoencoder_metrics:
                if metric in metrics:
                    print(f"{metric.upper():10}: {metrics[metric]:.4f}")
        
        # Print classification metrics if available
        classification_available = any(metric in metrics for metric in classification_metrics)
        if classification_available:
            print("\nClassification Metrics:")
            print("-" * 20)
            for metric in classification_metrics:
                if metric in metrics:
                    print(f"{metric.capitalize():10}: {metrics[metric]:.4f}")
        
        # Skip confusion matrix for autoencoder
        if not isinstance(y_true, np.ndarray) or y_true.ndim != 2:
            plot_confusion_matrix(y_true, y_pred, args.save_cm)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()