import pandas as pd
from sklearn.ensemble import IsolationForest
import h5py

# Load data
# Note: You may need to adjust the data loading and preprocessing based on your actual data structure
data = pd.read_csv('cleaned_data.csv')

# Preprocess data (example - adjust according to your needs)
# Here we're just using numeric columns as an example
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
X = data[numeric_cols]

# Train Isolation Forest model
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X)

# Save model to HDF5 file
with h5py.File('Isolation_Forest.h5', 'w') as h5file:
    # You might need a more sophisticated way to save sklearn models to HDF5
    # This is a basic example that might need adjustment
    h5file.create_dataset('model_params', data=str(model.get_params()))
    
print("Model Isolation Forest telah berhasil disimpan sebagai Isolation_Forest.h5")