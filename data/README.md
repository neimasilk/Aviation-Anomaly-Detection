# Data Directory

This directory contains all the data used in the Aviation Anomaly Detection project.

## Structure

- `raw/`: Contains the original, unmodified aviation communication data
  - Audio recordings
  - Transcripts
  - Metadata files
  
- `processed/`: Contains the preprocessed and feature-extracted data
  - Normalized features
  - Encoded labels
  - Training/validation/test splits

## Data Flow

1. Raw data is collected and stored in `raw/`
2. Preprocessing scripts transform raw data into processed features
3. Processed data is used for model training and evaluation

## Responsibilities

- **Data Preprocessing (Yesinka)**:
  - Implement data cleaning procedures
  - Extract relevant features
  - Ensure data quality and consistency
  - Document preprocessing steps

## Guidelines

- Keep raw data unchanged
- Document any data transformations
- Use consistent naming conventions
- Include metadata when applicable
- Backup data regularly