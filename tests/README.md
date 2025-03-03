# Tests Directory

This directory contains all test code for the Aviation Anomaly Detection project.

## Purpose

The tests in this directory ensure the reliability and correctness of the codebase through unit tests, integration tests, and system tests.

## Contents

This folder should contain:

- Unit tests for individual modules
- Integration tests for component interactions
- System tests for end-to-end functionality
- Test fixtures and mock data
- Test configuration files

## Expected Structure

- `unit/`: Tests for individual functions and classes
  - `preprocessing/`: Tests for preprocessing modules
  - `models/`: Tests for model implementations
  - `evaluation/`: Tests for evaluation metrics
  - `utils/`: Tests for utility functions
- `integration/`: Tests for component interactions
- `system/`: End-to-end tests
- `fixtures/`: Test data and mock objects
- `conftest.py`: Pytest configuration and fixtures

## Responsibility

All team members are responsible for writing tests for their respective components:
- **Yesinka**: Tests for preprocessing modules
- **Yosafat**: Tests for model implementations
- **Samuel**: Tests for evaluation metrics
- **All**: Tests for shared utilities

## Guidelines

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest as the testing framework
- Create meaningful test fixtures
- Mock external dependencies
- Include both positive and negative test cases
- Run tests before committing changes