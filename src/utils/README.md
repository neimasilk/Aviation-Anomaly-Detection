# Utils Directory

This directory contains utility functions and helper modules that are used across different components of the Aviation Anomaly Detection project.

## Purpose

The utilities in this directory provide common functionality that can be reused by preprocessing, modeling, and evaluation modules, promoting code reuse and consistency.

## Contents

This folder should contain:

- Logging utilities
- Configuration management
- Data validation helpers
- File I/O operations
- Performance monitoring tools
- Common data structures
- Error handling utilities

## Expected Modules

- `logger.py`: Centralized logging configuration
- `config.py`: Configuration loading and management
- `validators.py`: Data validation utilities
- `file_utils.py`: File handling operations
- `performance.py`: Performance monitoring utilities
- `visualization.py`: Common visualization functions

## Responsibility

All team members contribute to and maintain the utils directory:
- Add reusable functions that benefit multiple modules
- Document utility functions thoroughly
- Ensure backward compatibility when updating utilities
- Write comprehensive tests for utility functions

## Guidelines

- Keep utility functions focused and single-purpose
- Document all parameters, return values, and exceptions
- Include appropriate error handling
- Write unit tests for all utility functions
- Avoid dependencies on specific modules outside utils
- Maintain backward compatibility when possible