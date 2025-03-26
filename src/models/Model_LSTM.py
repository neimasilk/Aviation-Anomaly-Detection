import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the preprocessed TF-IDF features dataset
def load_data(file_path='e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/processed/tfidf_features.csv'):
    try:
        data = pd.read_csv(file_path)
        print("Dataset shape:", data.shape)
        print("Features:", data.columns[:-1].tolist())  # Assuming last column is target
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Preprocessed TF-IDF features not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading preprocessed data: {str(e)}")

# Prepare data for LSTM
def preprocess_data(data, test_size=0.2, time_steps=3):
    # Split features and target
    X = data.iloc[:, :-1].values  # All columns except last
    y = data.iloc[:, -1].values   # Last column is target
    
    # Convert target to binary integers
    y = y.astype(int)
    
    # Validate target values
    unique_classes = np.unique(y)
    print(f"Unique classes in target: {unique_classes}")
    
    # Handle single-class dataset (unsupervised anomaly detection)
    if len(unique_classes) == 1:
        print("Warning: Only one class found in target. Using unsupervised approach.")
        print("Generating synthetic anomalies for testing...")
        
        # Create synthetic anomalies (5% of data)
        num_samples = len(X)
        num_anomalies = int(0.05 * num_samples)
        
        # Generate random indices for anomalies
        anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
        y_synthetic = np.zeros_like(y)
        y_synthetic[anomaly_indices] = 1
        y = y_synthetic
        
        print(f"Created {num_anomalies} synthetic anomalies for training")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data for LSTM [samples, time_steps, features]
    X_reshaped = []
    y_reshaped = []
    
    for i in range(len(X_scaled) - time_steps):
        X_reshaped.append(X_scaled[i:i + time_steps])
        y_reshaped.append(y[i + time_steps])
    
    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=test_size, random_state=42, stratify=y_reshaped
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set anomalies: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.2f}%)")
    print(f"Test set anomalies: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, scaler

# Build the LSTM model
def build_model(input_shape, lstm_units=32, dropout_rate=0.2):  # Reduced LSTM units
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),  # Reduced dense layer units
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=16):  # Reduced epochs and batch size
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return history

# Evaluate the model and detect anomalies
def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Predict probabilities
    y_pred_proba = model.predict(X_test)
    
    # Convert to binary predictions based on threshold
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Normal', 'Anomaly'])
    plt.yticks([0, 1], ['Normal', 'Anomaly'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()
    
    return precision, recall, f1

# Find optimal threshold
def find_optimal_threshold(model, X_test, y_test, thresholds=np.arange(0.1, 0.9, 0.05)):
    results = []
    
    for threshold in thresholds:
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        results.append((threshold, precision, recall, f1))
        
        print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Plot the results
    results_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision')
    plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall')
    plt.plot(results_df['Threshold'], results_df['F1'], label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find threshold with best F1 score
    best_idx = results_df['F1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'Threshold']
    
    print(f"\nBest threshold: {best_threshold:.2f} with F1: {results_df.loc[best_idx, 'F1']:.4f}")
    
    return best_threshold

# Save the model
def save_model(model, model_path='e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/model-output/lstm_anomaly_detector.h5'):
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Main function
def main():
    print("Loading data...")
    data = load_data()
    
    print("Preprocessing data...")
    time_steps = 3  # Reduced time steps
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, time_steps=time_steps)
    
    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, lstm_units=32, dropout_rate=0.2)  # Using smaller model
    model.summary()
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=16)
    
    # Save the trained model
    print("Saving model...")
    save_model(model)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Finding optimal threshold...")
    best_threshold = find_optimal_threshold(model, X_test, y_test)
    
    print("Evaluating model with optimal threshold...")
    precision, recall, f1 = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    
    print(f"Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()