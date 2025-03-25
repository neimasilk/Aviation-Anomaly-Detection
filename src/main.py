#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Aviation Anomaly Detection system.
This module orchestrates the data preprocessing, model training/loading,
and anomaly detection processes.
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the anomaly detection system.
    """
    logger.info("Starting Aviation Anomaly Detection System")
    
    try:
        # Load and preprocess data
        from models.Model_Autoencoder import AutoencoderAnomalyDetector
        import pandas as pd
        
        # Load data
        data_path = 'e:/Kuliah/PKL LabDataScience/Aviation-Anomaly-Detection/data/processed/cleaned_data.csv'
        df = pd.read_csv(data_path)
        features = df.select_dtypes(include=['float64', 'int64']).columns
        X = df[features].values
        
        # Initialize model
        input_dim = X.shape[1]
        model = AutoencoderAnomalyDetector(input_dim=input_dim)
        
        # Split data and train model
        train_loader, val_loader, test_loader = model.split_data(X)
        model.train(train_loader, validation_loader=val_loader)
        
        # Compute threshold
        threshold = model.compute_threshold(train_loader)
        logger.info(f"Anomaly threshold: {threshold:.6f}")
        
        # Evaluate model
        test_loss, accuracy, precision = model.evaluate(test_loader)
        logger.info(f"Test Loss: {test_loss:.6f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1
    
    logger.info("Aviation Anomaly Detection System completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())