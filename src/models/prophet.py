#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prophet model for currency prediction.

This module implements a Prophet model for predicting currency exchange rates
using Facebook's Prophet library.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore', module='prophet')


class ProphetModel:
    """
    Prophet model for currency prediction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Prophet model.
        
        Args:
            **kwargs: Arguments to pass to Prophet model
        """
        self.model = Prophet(**kwargs)
        self.data = None
        self.prophet_data = None
        self.test_size = 30  # Default test size
        
    def fit(self, data, test_size=None):
        """
        Fit the Prophet model to the data.
        
        Args:
            data (pandas.DataFrame): Data with DatetimeIndex and 'Close' column
            test_size (int, optional): Number of data points to use for testing
            
        Returns:
            self: The fitted model instance
        """
        if data is None or data.empty:
            raise ValueError("No data provided")
            
        self.data = data.copy()
        
        if test_size is not None:
            self.test_size = test_size
        
        # Convert data to Prophet format (ds, y)
        self.prophet_data = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data['Close']
        })
        
        # Split data into train and test sets
        self.train = self.prophet_data.iloc[:-self.test_size]
        self.test = self.prophet_data.iloc[-self.test_size:]
        
        # Fit the model on the training data
        try:
            self.model.fit(self.train)
            return self
        except Exception as e:
            raise Exception(f"Failed to fit Prophet model: {str(e)}")
    
    def predict(self, steps):
        """
        Make predictions for future time steps.
        
        Args:
            steps (int): Number of steps to predict
            
        Returns:
            pandas.DataFrame: DataFrame with DatetimeIndex and predictions
        """
        if not hasattr(self.model, 'history'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create a dataframe for future dates
        future = self.model.make_future_dataframe(periods=steps)
        
        # Make predictions
        prophet_forecast = self.model.predict(future)
        
        # Get only the future predictions
        forecast = prophet_forecast.iloc[-steps:]
        
        # Convert back to the original format with DatetimeIndex
        predictions = pd.DataFrame({
            'Close': forecast['yhat'],
            'Lower': forecast['yhat_lower'],
            'Upper': forecast['yhat_upper']
        }, index=pd.DatetimeIndex(forecast['ds']))
        
        return predictions
    
    def evaluate(self):
        """
        Evaluate the model performance on the test set.
        
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not hasattr(self.model, 'history'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create a dataframe for the test period
        future = pd.DataFrame({'ds': self.test['ds']})
        
        # Make predictions for the test period
        prophet_forecast = self.model.predict(future)
        
        # Calculate metrics
        mse = mean_squared_error(self.test['y'], prophet_forecast['yhat'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.test['y'], prophet_forecast['yhat'])
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }


def test_prophet():
    """
    Test function for the Prophet model.
    """
    # Create some sample data
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')
    np.random.seed(42)
    values = np.cumsum(np.random.normal(0, 0.1, len(dates))) + 20
    
    # Create DataFrame
    data = pd.DataFrame({'Close': values}, index=dates)
    
    # Create and fit model
    model = ProphetModel()
    model.fit(data, test_size=30)
    
    # Evaluate model
    metrics = model.evaluate()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Make predictions
    predictions = model.predict(steps=15)
    print("\nPredictions:")
    print(predictions.head())


if __name__ == "__main__":
    test_prophet()