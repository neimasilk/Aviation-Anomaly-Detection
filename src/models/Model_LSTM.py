import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the preprocessed data with memory optimization
def load_data(file_path='e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/processed/cleaned_data.csv'):
    try:
        # Use low_memory=False for better performance with large files
        data = pd.read_csv(file_path, low_memory=False)
        print("Dataset shape:", data.shape)
        
        # Check if text column exists (assuming 'text' or 'message' column contains communication data)
        text_columns = [col for col in data.columns if col.lower() in ['text', 'message', 'communication', 'transcript', 'cleaned_text', 'normalized_text']]
        if not text_columns:
            raise ValueError("No text column found for flight communications")
        
        text_column = text_columns[0]
        print(f"Using '{text_column}' as the text data column")
        
        # Check if 'anomaly' column exists, if not, create it
        if 'anomaly' not in data.columns:
            print("No 'anomaly' column found. Creating synthetic anomaly labels for unsupervised learning.")
            
            # For demonstration, we'll mark approximately 5% of the data as anomalies
            num_samples = len(data)
            num_anomalies = int(0.05 * num_samples)
            
            # Generate random indices for anomalies
            np.random.seed(42)  # For reproducibility
            anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
            
            # Create anomaly column (0 = normal, 1 = anomaly)
            data['anomaly'] = 0
            data.loc[anomaly_indices, 'anomaly'] = 1
            
            print(f"Created synthetic anomaly labels: {num_anomalies} anomalies ({num_anomalies/num_samples*100:.2f}%)")
        
        # Move anomaly column to the end if it's not already there
        if data.columns[-1] != 'anomaly':
            anomaly_col = data.pop('anomaly')
            data['anomaly'] = anomaly_col
            
        # Keep only necessary columns to reduce memory usage
        keep_cols = [text_column, 'anomaly']
        data = data[keep_cols]
        
        print("Features:", data.columns[:-1].tolist())
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Cleaned data not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# Prepare text data for LSTM with optimized memory usage
def preprocess_data(data, test_size=0.2, max_sequence_length=100, max_words=10000):
    # Identify text column
    text_columns = [col for col in data.columns if col.lower() in ['text', 'message', 'communication', 'transcript', 'cleaned_text', 'normalized_text']]
    text_column = text_columns[0] if text_columns else None
    
    # Get text data and target more efficiently
    if text_column:
        X_text = data[text_column].fillna('').astype(str).values
    else:
        X_text = data.iloc[:, :-1].astype(str).apply(lambda x: ' '.join(x), axis=1).values
    
    y = data.iloc[:, -1].values.astype(int)
    
    # Validate target values
    unique_classes = np.unique(y)
    print(f"Unique classes in target: {unique_classes}")
    
    # Tokenize text with optimized settings
    tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(X_text)
    
    # Convert text to sequences
    X_sequences = tokenizer.texts_to_sequences(X_text)
    
    # Pad sequences to ensure uniform length
    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Free up memory
    del X_sequences
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Free up more memory
    del X_padded
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set anomalies: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.2f}%)")
    print(f"Test set anomalies: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, tokenizer, max_sequence_length

# Build an optimized LSTM model for text data
def build_model(vocab_size, max_sequence_length, embedding_dim=100, lstm_units=64, dropout_rate=0.3):
    model = Sequential([
        # Add embedding layer with optimized settings
        Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, 
                 input_length=max_sequence_length, mask_zero=True),
        
        # Use Bidirectional LSTM for better context understanding
        Bidirectional(LSTM(lstm_units, return_sequences=True, 
                          recurrent_dropout=0.0, unroll=False)),  # Disable recurrent dropout for better performance
        Dropout(dropout_rate),
        
        # Second LSTM layer
        Bidirectional(LSTM(lstm_units // 2, return_sequences=False)),
        Dropout(dropout_rate),
        
        # Dense layers with optimized sizes
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    # Use Adam optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Train the model with early stopping and checkpointing
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    # Define callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath='e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/model-output/lstm_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train with optimized settings and memory-efficient batch size
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history

# Evaluate model performance with memory optimization
def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Get predictions in batches to save memory
    batch_size = 128
    y_pred_proba = model.predict(X_test, batch_size=batch_size)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Print detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return precision, recall, f1

# Find optimal threshold for classification with memory optimization
def find_optimal_threshold(model, X_test, y_test):
    # Get prediction probabilities in batches
    batch_size = 128
    y_pred_proba = model.predict(X_test, batch_size=batch_size)
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    print("\nFinding optimal threshold:")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nOptimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold

# Save the trained model
def save_model(model, path='e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/model-output/lstm_model.h5'):
    model.save(path)
    print(f"Model saved to {path}")

# Main function with optimized workflow
def main():
    # Enable memory growth for GPU to prevent OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
    
    print("Loading data...")
    data = load_data()
    
    print("Preprocessing text data...")
    X_train, X_test, y_train, y_test, tokenizer, max_sequence_length = preprocess_data(
        data, max_sequence_length=100, max_words=10000
    )
    
    print("Building model...")
    vocab_size = len(tokenizer.word_index)
    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=100,
        lstm_units=64
    )
    model.summary()
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
    
    # Save the trained model
    print("Saving model...")
    save_model(model)
    
    print("Finding optimal threshold...")
    best_threshold = find_optimal_threshold(model, X_test, y_test)
    
    print("Evaluating model with optimal threshold...")
    precision, recall, f1 = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    
    print(f"Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()