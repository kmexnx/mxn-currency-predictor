#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing functionality for currency data.

This module provides tools for cleaning, transforming, and preparing currency
data for prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    """
    Class for preprocessing currency data.
    """
    
    def __init__(self):
        """
        Initialize the DataProcessor.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = None
        
    def process(self, data, fill_missing=True, add_features=True):
        """
        Process the raw currency data.
        
        Args:
            data (pandas.DataFrame): Raw currency data with DatetimeIndex and 'Close' column
            fill_missing (bool): Whether to fill missing values
            add_features (bool): Whether to add additional features
            
        Returns:
            pandas.DataFrame: Processed data
        """
        if data is None or data.empty:
            return None
        
        # Make a copy to avoid modifying the original data
        processed = data.copy()
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(processed.index, pd.DatetimeIndex):
            try:
                processed.index = pd.to_datetime(processed.index)
            except:
                raise ValueError("Index could not be converted to DatetimeIndex")
        
        # Sort by date
        processed = processed.sort_index()
        
        # Fill missing values if requested
        if fill_missing:
            processed = self._fill_missing_values(processed)
        
        # Add additional features if requested
        if add_features:
            processed = self._add_features(processed)
        
        return processed
    
    def _fill_missing_values(self, data):
        """
        Fill missing values in the data.
        
        Args:
            data (pandas.DataFrame): Data with possible missing values
            
        Returns:
            pandas.DataFrame: Data with missing values filled
        """
        # Check for missing values
        if data['Close'].isnull().sum() > 0:
            # Forward fill (use previous day's value)
            data = data.fillna(method='ffill')
            
            # If there are still missing values (e.g., at the beginning), use backward fill
            if data['Close'].isnull().sum() > 0:
                data = data.fillna(method='bfill')
        
        return data
    
    def _add_features(self, data):
        """
        Add additional features to the data.
        
        Args:
            data (pandas.DataFrame): Currency data
            
        Returns:
            pandas.DataFrame: Data with additional features
        """
        # Add rolling mean features
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # Add volatility features
        data['Volatility5'] = data['Close'].rolling(window=5).std()
        data['Volatility10'] = data['Close'].rolling(window=10).std()
        
        # Add price change features
        data['PctChange'] = data['Close'].pct_change()
        data['PctChange5'] = data['Close'].pct_change(periods=5)
        
        # Add day of week (0=Monday, 6=Sunday)
        data['DayOfWeek'] = data.index.dayofweek
        
        # Add month
        data['Month'] = data.index.month
        
        # Fill NaN values created by rolling calculations
        data = data.fillna(method='bfill')
        
        return data
    
    def scale_data(self, data, column='Close'):
        """
        Scale the data using MinMaxScaler.
        
        Args:
            data (pandas.DataFrame): Data to scale
            column (str): Column to scale
            
        Returns:
            numpy.ndarray: Scaled data
        """
        values = data[column].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(values)
        return self.scaled_data
    
    def inverse_scale(self, scaled_values):
        """
        Inverse transform scaled values back to original scale.
        
        Args:
            scaled_values (numpy.ndarray): Scaled values
            
        Returns:
            numpy.ndarray: Values in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call scale_data first.")
            
        # Ensure scaled_values is 2D
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
            
        return self.scaler.inverse_transform(scaled_values)
    
    def create_sequences(self, data, seq_length):
        """
        Create sequences for time series models like LSTM.
        
        Args:
            data (numpy.ndarray): Input data array
            seq_length (int): Sequence length
            
        Returns:
            tuple: (X, y) where X is the sequences and y is the target values
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
            
        return np.array(X), np.array(y)


def test_processor():
    """
    Test function for the DataProcessor.
    """
    # Create some sample data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31')
    values = np.random.normal(20, 0.5, len(dates))
    values = np.cumsum(np.random.normal(0, 0.1, len(dates))) + values
    
    # Create DataFrame
    data = pd.DataFrame({'Close': values}, index=dates)
    
    # Create processor
    processor = DataProcessor()
    
    # Process data
    processed = processor.process(data)
    
    print("Original data:")
    print(data.head())
    
    print("\nProcessed data:")
    print(processed.head())
    
    # Test scaling
    scaled = processor.scale_data(processed)
    print("\nScaled data (first 5 values):")
    print(scaled[:5])
    
    # Test inverse scaling
    original = processor.inverse_scale(scaled[:5])
    print("\nInverse scaled:")
    print(original)
    
    # Test sequence creation
    X, y = processor.create_sequences(scaled, seq_length=5)
    print("\nSequence shape:", X.shape)
    print("Target shape:", y.shape)


if __name__ == "__main__":
    test_processor()