import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from metrics import load_model_from_file, load_and_preprocess_data, calculate_metrics

def plot_metrics(metrics, save_path=None):
    """Plot bar graph showing model metrics.
    
    Args:
        metrics (dict): Dictionary containing model metrics
        save_path (str, optional): Path to save the plot
    """
    try:
        # Only show classification metrics
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        labels = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']

        values = [metrics.get(m, 0) for m in metric_names]
        x = np.arange(len(metric_names))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, values, width=0.6)

        ax.set_ylabel('Nilai')
        ax.set_title('Metrik Model')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")

def plot_error_heatmap(y_true, y_pred, feature_names=None, save_path=None):
    """Plot heatmap showing areas where model makes mistakes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        feature_names (list, optional): Names of features
        save_path (str, optional): Path to save the plot
    """
    try:
        # Calculate error matrix
        errors = np.abs(y_true - y_pred)
        if errors.ndim > 1:
            error_matrix = np.mean(errors, axis=0).reshape(-1, 1)
        else:
            error_matrix = errors.reshape(-1, 1)

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(error_matrix))]

        # Create heatmap
        plt.figure(figsize=(8, 10))
        sns.heatmap(error_matrix, 
                    annot=True, 
                    cmap='YlOrRd', 
                    yticklabels=feature_names,
                    xticklabels=['Error Rate'])
        plt.title('Area Kesalahan Model')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Heatmap saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error plotting error heatmap: {str(e)}")

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve showing model's ability to distinguish between normal and anomaly cases.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path (str, optional): Path to save the plot
    """
    try:
        # Handle autoencoder case (reconstruction error)
        if y_true.ndim == 2 and y_true.shape == y_pred_proba.shape:
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(y_true - y_pred_proba), axis=1)
            
            # Create binary labels based on reconstruction error threshold
            # Calculate threshold using percentile method consistent with metrics.py
            threshold = np.percentile(reconstruction_error, 95)
            
            # Validate reconstruction error input
            if reconstruction_error.ndim != 1:
                reconstruction_error = reconstruction_error.flatten()
            
            # Generate labels based on original data if available
            if isinstance(y_true, np.ndarray) and y_true.ndim == 1:
                y_true_binary = y_true
            else:
                y_true_binary = (reconstruction_error > threshold).astype(int)
            
            # Calculate anomaly score with normalization
            anomaly_score = (reconstruction_error - np.min(reconstruction_error)) / \
            (np.max(reconstruction_error) - np.min(reconstruction_error))
            fpr, tpr, _ = roc_curve(y_true_binary, anomaly_score)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
            
        # Handle multi-class case
        elif y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 2:
            n_classes = y_pred_proba.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Convert y_true to one-hot encoding
            y_true_bin = pd.get_dummies(y_true).values

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure(figsize=(10, 8))
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))

            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'Kelas {i} (AUC = {roc_auc[i]:.2f})')

        else:
            # Binary classification case
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path)
            print(f"ROC curve saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error plotting ROC curve: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize autoencoder model performance')
    parser.add_argument('--model', required=True, help='Path to the autoencoder model (.h5 file)')
    parser.add_argument('--data', required=True, help='Path to the test data (.csv file)')
    parser.add_argument('--save-dir', help='Directory to save plots')
    
    args = parser.parse_args()
    
    try:
        # Create save paths if save_dir is provided
        if args.save_dir:
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            metrics_path = os.path.join(args.save_dir, 'model_metrics.png')
            heatmap_path = os.path.join(args.save_dir, 'error_heatmap.png')
            roc_path = os.path.join(args.save_dir, 'roc_curve.png')
        else:
            metrics_path = heatmap_path = roc_path = None

        # Load model and data
        print("\nLoading model and data...")
        model = load_model_from_file(args.model)
        X, y_true = load_and_preprocess_data(args.data)

        # Get predictions
        print("\nGenerating predictions...")
        y_pred = model.predict(X)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)

        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Plot metrics
        print("\n1. Plotting model metrics...")
        plot_metrics(metrics, metrics_path)

        # Plot error heatmap
        print("\n2. Plotting error heatmap...")
        plot_error_heatmap(y_true, y_pred, save_path=heatmap_path)

        # Plot ROC curve
        print("\n3. Plotting ROC curve...")
        plot_roc_curve(y_true, y_pred, save_path=roc_path)

        print("\nVisualization complete!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0

if __name__ == '__main__':
    main()