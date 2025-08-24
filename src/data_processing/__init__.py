"""
Data Processing Module for ICU Patient Outcome Prediction

This module provides comprehensive data processing capabilities for ICU patient data,
including cohort definition, static and time-series data extraction, preprocessing,
and train/validation/test split creation for machine learning models.

The module is organized into several components:
- Cohort definition and target label creation
- Static patient data extraction and preprocessing  
- Time-series vital signs and lab values processing
- Integrated preprocessing pipeline
- Utility functions for logging and data management

Main workflow:
1. Define patient cohort with inclusion/exclusion criteria
2. Extract static patient features (demographics, treatments)
3. Extract time-series features (vitals, labs) with temporal aggregation
4. Apply preprocessing (encoding, imputation, scaling)
5. Create stratified train/validation/test splits
"""

