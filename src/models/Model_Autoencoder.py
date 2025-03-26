import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import logging
from typing import Tuple, List, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Add this at the top with other imports
import h5py

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):  # Reduced encoding dimension for stronger bottleneck
        super(Autoencoder, self).__init__()
        
        # Enhanced Encoder with more layers and lower dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),  # LeakyReLU for better gradient flow
            nn.Dropout(0.1),    # Reduced dropout to preserve more information
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, encoding_dim),
            nn.Tanh()  # Tanh for better feature distribution in latent space
        )
        
        # Enhanced Decoder with more layers and lower dropout
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim: int, encoding_dim: int = 64, learning_rate: float = 0.0001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = Autoencoder(input_dim, encoding_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.threshold = None
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.actual_epochs = 0  # Menambahkan atribut tracking epoch
        self.scaler = StandardScaler()
        
        # Validasi parameter komunikasi penerbangan
        self.min_freq = 118.0  # MHz (VHF band bawah)
        self.max_freq = 137.0  # MHz (VHF band atas)
        self.valid_modes = ['AM', 'FM', 'USB', 'LSB']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_loader: DataLoader, num_epochs: int = 100, patience: int = 5,
              validation_loader: Optional[DataLoader] = None) -> int:
        try:
            self.model.train()
            min_loss = float('inf')
            patience_counter = 0
            
            for epoch in tqdm(range(num_epochs), desc="Training"):
                total_loss = 0
                batch_count = 0
                
                for batch_x in train_loader:
                    batch_x = batch_x[0].to(self.device)
                    
                    # Clear gradients
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_x)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                avg_loss = total_loss / batch_count
                
                # Validation if provided
                if validation_loader:
                    val_loss = self._validate(validation_loader)
                    self.logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')
                    
                    # Early stopping check
                    if val_loss < min_loss:
                        min_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        self.actual_epochs = epoch + 1
                        self.logger.info(f'Early stopping triggered after {self.actual_epochs} epochs')
                        break
                else:
                    self.logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
            
            # Update actual epochs jika tidak early stopping
            if not validation_loader or self.actual_epochs == 0:
                self.actual_epochs = num_epochs
                
            # Memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return self.actual_epochs
                
        except Exception as e:
            self.logger.error(f'Error during training: {str(e)}')
            raise
    
    def _validate(self, validation_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_x in validation_loader:
                batch_x = batch_x[0].to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                total_loss += loss.item()
                batch_count += 1
                
        return total_loss / batch_count
    
    def compute_threshold(self, train_loader: DataLoader, percentile: float = 95) -> float:
        try:
            self.model.eval()
            reconstruction_errors = []
            
            with torch.no_grad():
                for batch_x in tqdm(train_loader, desc="Computing threshold"):
                    batch_x = batch_x[0].to(self.device)
                    outputs = self.model(batch_x)
                    errors = torch.mean((outputs - batch_x) ** 2, dim=1)
                    reconstruction_errors.extend(errors.cpu().numpy())
            
            self.threshold = np.percentile(reconstruction_errors, percentile)
            self.logger.info(f'Computed anomaly threshold: {self.threshold:.6f}')
            return self.threshold
            
        except Exception as e:
            self.logger.error(f'Error computing threshold: {str(e)}')
            raise
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if self.threshold is None:
                raise ValueError("Threshold not computed. Run compute_threshold first.")
            
            # Validasi input spesifik aviation
            self._validate_aviation_input(x)
                
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(x).to(self.device)
                outputs = self.model(x)
                errors = torch.mean((outputs - x) ** 2, dim=1)
                anomalies = (errors > self.threshold).cpu().numpy()
                
                return anomalies, errors.cpu().numpy()
                
        except Exception as e:
            self.logger.error(f'Error during prediction: {str(e)}')
            raise
    
    def _validate_aviation_input(self, x: np.ndarray) -> None:
        """Validasi input spesifik domain komunikasi penerbangan"""
        try:
            # Cek frekuensi dalam range VHF aviation
            freq_features = x[:, 1]  # Asumsi fitur frekuensi di kolom kedua
            if np.any((freq_features < self.min_freq) | (freq_features > self.max_freq)):
                invalid_count = np.sum((freq_features < self.min_freq) | (freq_features > self.max_freq))
                self.logger.warning(f'Deteksi {invalid_count} sampel dengan frekuensi di luar range VHF aviation')
            
            # Cek pola transmisi valid
            mode_features = x[:, 3].astype(int)  # Asumsi fitur mode di kolom keempat
            invalid_modes = np.isin(mode_features, [0,1,2,3], invert=True)
            if np.any(invalid_modes):
                self.logger.error('Deteksi mode transmisi tidak valid dalam input data')
                raise ValueError('Mode transmisi tidak valid terdeteksi')
            
            # Cek nilai non-negatif untuk fitur tertentu
            if np.any(x[:, [0,2,4]] < 0):  # Asumsi fitur power, duration, error_rate
                self.logger.error('Deteksi nilai negatif pada fitur yang harus non-negatif')
                raise ValueError('Nilai negatif terdeteksi dalam fitur input')
                
        except Exception as e:
            self.logger.error(f'Validasi input gagal: {str(e)}')
            raise
    
    
    def save_model(self, filepath: str = 'e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/Model-output/autoencoder_model.h5') -> None:
        try:
            # Create directory structure if not exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Delete existing file if present
            if os.path.exists(filepath):
                os.remove(filepath)
                
            # Save model state and threshold
            with h5py.File(filepath, 'w') as f:
                # Save model architecture and weights
                model_group = f.create_group('model')
                for name, param in self.model.state_dict().items():
                    model_group.create_dataset(name, data=param.cpu().numpy())
                
                # Save threshold and metadata
                f.create_dataset('threshold', data=self.threshold)
                f.attrs['input_dim'] = self.model.encoder[0].in_features
                f.attrs['encoding_dim'] = self.model.encoder[-2].out_features
                
            self.logger.info(f'Model successfully saved to {filepath}')
            
        except Exception as e:
            self.logger.error(f'Error saving model: {str(e)}')
            raise
    
    def load_model(self, filepath: str = 'e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/Model-output/autoencoder_model.h5') -> None:
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
                
            with h5py.File(filepath, 'r') as f:
                # Load model parameters
                model_state = {}
                for param_name in f['model']:
                    param_data = f['model'][param_name][()]
                    model_state[param_name] = torch.from_numpy(param_data)
                
                # Load threshold and verify dimensions
                self.threshold = f['threshold'][()]
                self.model.load_state_dict(model_state)
                
                # Verify model architecture matches
                input_dim = f.attrs['input_dim']
                encoding_dim = f.attrs['encoding_dim']
                if (self.model.encoder[0].in_features != input_dim or 
                    self.model.encoder[-2].out_features != encoding_dim):
                    raise ValueError("Loaded model architecture doesn't match current configuration")
                    
            self.logger.info(f'Model successfully loaded from {filepath}')
            
        except Exception as e:
            self.logger.error(f'Error loading model: {str(e)}')
            raise

    def preprocess_aviation_data(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocessing khusus data komunikasi penerbangan:
        1. Validasi kolom yang diperlukan
        2. Handling missing values
        3. Encoding fitur kategorikal
        4. Normalisasi fitur
        """
        try:
            # Validasi kolom input
            required_columns = ['Power', 'Frequency', 'Duration', 'Mode', 'Error_Rate']
            if not set(required_columns).issubset(raw_data.columns):
                missing = set(required_columns) - set(raw_data.columns)
                raise ValueError(f'Kolom yang diperlukan tidak ditemukan: {missing}')

            # Handling missing values
            data_clean = raw_data[required_columns].dropna()
            
            # Encoding mode transmisi
            mode_mapping = {'AM': 0, 'FM': 1, 'USB': 2, 'LSB': 3}
            data_clean['Mode'] = data_clean['Mode'].map(mode_mapping).astype(int)
            
            # Normalisasi fitur
            X = self.scaler.fit_transform(data_clean)
            
            # Validasi range frekuensi
            freq_idx = required_columns.index('Frequency')
            if np.any((X[:, freq_idx] < self.min_freq) | (X[:, freq_idx] > self.max_freq)):
                self.logger.warning('Terdapat frekuensi di luar range valid setelah normalisasi')
            
            return X
            
        except Exception as e:
            self.logger.error(f'Gagal preprocessing data: {str(e)}')
            raise

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        try:
            # Validasi input DataFrame
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input harus berupa pandas DataFrame")
                
            X = df.select_dtypes(include=['float64', 'int64']).values
            X = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
            X_train, X_val = train_test_split(X_train, test_size=val_size/(1-test_size), random_state=42)
            
            # Convert to TensorDataset
            train_dataset = TensorDataset(torch.FloatTensor(X_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test))
            
            return (
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                DataLoader(val_dataset, batch_size=batch_size),
                DataLoader(test_dataset, batch_size=batch_size)
            )
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, float, float, float]:
        try:
            if self.threshold is None:
                raise ValueError("Threshold belum dihitung. Jalankan compute_threshold terlebih dahulu")
            
            self.model.eval()
            y_true = []
            y_scores = []
            
            with torch.no_grad():
                for batch_x in test_loader:
                    batch_x = batch_x[0].to(self.device)
                    outputs = self.model(batch_x)
                    errors = torch.mean((outputs - batch_x)**2, dim=1)
                    y_scores.extend(errors.cpu().numpy())
                    y_true.extend(np.zeros(len(errors)))  # Asumsi semua data test normal
            
            y_pred = (np.array(y_scores) > self.threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            
            return np.mean(y_scores), accuracy, precision, recall, f1
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        df = pd.read_csv('e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/processed/cleaned_data.csv')
        
        input_dim = df.select_dtypes(include=['float64', 'int64']).shape[1]
        model = AutoencoderAnomalyDetector(input_dim=input_dim)
        
        train_loader, val_loader, test_loader = model.split_data(df)
        
        model.train(train_loader, validation_loader=val_loader)
        
        threshold = model.compute_threshold(train_loader)
        print(f"\nAnomaly threshold: {threshold:.6f}")
        
        test_loss, accuracy, precision, recall, f1_score = model.evaluate(test_loader)
        
        # Save the trained model
        model.save_model()
        
        print(f"\nTest Loss: {test_loss:.6f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
    except Exception as e:
        model.logger.error(f"Error during execution: {str(e)}")
        raise
    