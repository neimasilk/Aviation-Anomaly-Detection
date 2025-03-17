import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load breast cancer dataset instead of cleaned_data.csv
data = pd.read_csv('breast-cancer-wisconsin.csv')
# Handle missing values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and target (assuming last column is the target)
X = data.iloc[:, 1:-1].astype(float)  # Convert to float explicitly
y = data.iloc[:, -1].astype(int)  # Convert to int explicitly

# Convert target to binary (4 for malignant)
y = (y == 4).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

# Save model to H5 file
model.save('model.h5')

# Load saved model
loaded_model = load_model('model.h5')

# Make predictions
predictions = loaded_model.predict(X_test_scaled)
y_pred = (predictions > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, predictions)

# Visualization - Metrics Bar Chart
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

plt.figure(figsize=(8,5))
sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
plt.title('Model Evaluation Metrics Comparison')
plt.ylim(0, 1)
plt.savefig('static/metrics_barplot.png')
plt.close()

# Visualization - Feature Correlation Heatmap
plt.figure(figsize=(10,8))
corr_matrix = pd.DataFrame(X_train).corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('static/feature_heatmap.png')
plt.close()

# Visualization - ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, predictions)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('static/roc_curve.png')
plt.close()

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')

# Build autoencoder model
def build_autoencoder_model(input_dim):
    model = Sequential([
        # Encoder
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        
        # Decoder
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear')  # Output layer for reconstruction
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',  # Mean Squared Error for autoencoder
                  metrics=['mae'])
    return model

# Build classification model
def build_classification_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Model comparison function
def compare_models():
    input_dim = X_train_scaled.shape[1]
    
    # Train and evaluate classification model
    clf_model = build_classification_model(input_dim)
    clf_history = clf_model.fit(X_train_scaled, y_train,
                               epochs=50, batch_size=32,
                               validation_split=0.2, verbose=0)
    clf_model.save('model.h5')
    clf_preds = clf_model.predict(X_test_scaled)
    
    # Train and evaluate autoencoder
    autoencoder = build_autoencoder_model(input_dim)
    autoencoder_history = autoencoder.fit(X_train_scaled, X_train_scaled,
                                        epochs=50, batch_size=32,
                                        validation_split=0.2, verbose=0)
    autoencoder.save('autoencoder_model.h5')
    reconstructions = autoencoder.predict(X_test_scaled)
    
    # Calculate reconstruction metrics
    mse = np.mean(np.square(X_test_scaled - reconstructions))
    mae = np.mean(np.abs(X_test_scaled - reconstructions))
    
    return {
        'Classification': {
            'accuracy': accuracy_score(y_test, (clf_preds > 0.5).astype(int)),
            'roc_auc': roc_auc_score(y_test, clf_preds),
            'params': clf_model.count_params()
        },
        'Autoencoder': {
            'mse': float(mse),  # Convert numpy float to Python float
            'mae': float(mae),  # Convert numpy float to Python float
            'params': autoencoder.count_params()
        }
    }

# Visualize comparison
def visualize_comparison(comparison):
    plt.figure(figsize=(12, 6))
    
    # Classification metrics
    plt.subplot(1, 2, 1)
    sns.barplot(x=['Accuracy', 'ROC-AUC'],
                y=[comparison['Classification']['accuracy'], comparison['Classification']['roc_auc']])
    plt.title('Classification Model Performance')
    plt.ylim(0, 1)
    
    # Autoencoder metrics
    plt.subplot(1, 2, 2)
    sns.barplot(x=['MSE', 'MAE'],
                y=[comparison['Autoencoder']['mse'], comparison['Autoencoder']['mae']])
    plt.title('Autoencoder Reconstruction Error')
    
    plt.tight_layout()
    plt.savefig('static/model_comparison.png')
    plt.close()

# Run model comparison if this file is executed directly
if __name__ == '__main__':
    # Model comparison
    model_comparison = compare_models()
    visualize_comparison(model_comparison)
    
    # Print results
    print("\n=== Classification Model ===")
    print(f"Accuracy: {model_comparison['Classification']['accuracy']:.4f}")
    print(f"ROC-AUC: {model_comparison['Classification']['roc_auc']:.4f}")
    print(f"Parameters: {model_comparison['Classification']['params']}")
    
    print("\n=== Autoencoder Model ===")
    print(f"MSE: {model_comparison['Autoencoder']['mse']:.4f}")
    print(f"MAE: {model_comparison['Autoencoder']['mae']:.4f}")
    print(f"Parameters: {model_comparison['Autoencoder']['params']}")


@tf.function(reduce_retracing=True)
def train_step(model, optimizer, x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = model.compiled_loss(y_batch, predictions)
    
    # Explicit type handling for gradients
    gradients: list[tf.Tensor | None] = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        list(zip(gradients, model.trainable_variables))
    )
    return loss

# Gunakan dalam training loop (Example usage in model training)
# for epoch in range(epochs):
#     for x_batch, y_batch in train_dataset:
#         loss = train_step(model, optimizer, x_batch, y_batch)