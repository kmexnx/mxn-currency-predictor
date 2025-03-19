#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARIMA model for currency prediction.

This module implements an ARIMA (AutoRegressive Integrated Moving Average) model
for predicting currency exchange rates.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.simplefilter('ignore', ConvergenceWarning)


class ARIMAModel:
    """
    ARIMA model for currency prediction.
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize the ARIMA model.
        
        Args:
            order (tuple): ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.results = None
        self.data = None
        self.test_size = 30  # Default test size
        
    def fit(self, data, test_size=None):
        """
        Fit the ARIMA model to the data.
        
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
        
        # Split data into train and test sets
        self.train = self.data.iloc[:-self.test_size]
        self.test = self.data.iloc[-self.test_size:]
        
        # Fit the model on the training data
        try:
            self.model = ARIMA(self.train['Close'], order=self.order)
            self.results = self.model.fit()
            return self
        except Exception as e:
            print(f"Error fitting ARIMA model: {str(e)}")
            # Try with a different order if the original fails
            try:
                print("Trying with order (1,1,0)...")
                self.order = (1, 1, 0)
                self.model = ARIMA(self.train['Close'], order=self.order)
                self.results = self.model.fit()
                return self
            except Exception as e2:
                raise Exception(f"Failed to fit ARIMA model: {str(e2)}")
    
    def predict(self, steps):
        """
        Make predictions for future time steps.
        
        Args:
            steps (int): Number of steps to predict
            
        Returns:
            pandas.DataFrame: DataFrame with DatetimeIndex and predictions
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the last date in the data
        last_date = self.data.index[-1]
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Make the forecast
        forecast = self.results.forecast(steps=steps)
        
        # Create the prediction DataFrame
        predictions = pd.DataFrame({
            'Close': forecast,
            'Lower': np.nan,  # Placeholder for lower confidence interval
            'Upper': np.nan   # Placeholder for upper confidence interval
        }, index=future_dates)
        
        try:
            # Try to get prediction intervals
            pred_int = self.results.get_prediction(start=len(self.train), end=len(self.train) + steps - 1)
            conf_int = pred_int.conf_int()
            
            # Set the confidence intervals if available
            if not conf_int.empty and len(conf_int) >= len(predictions):
                predictions['Lower'] = conf_int.iloc[:len(predictions), 0].values
                predictions['Upper'] = conf_int.iloc[:len(predictions), 1].values
        except:
            # If getting intervals fails, just continue without them
            pass
        
        return predictions
    
    def evaluate(self):
        """
        Evaluate the model performance on the test set.
        
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Make predictions on the test set
        predictions = self.results.forecast(steps=len(self.test))
        
        # Calculate metrics
        mse = mean_squared_error(self.test['Close'], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.test['Close'], predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }


def test_arima():
    """
    Test function for the ARIMA model.
    """
    # Create some sample data
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')
    np.random.seed(42)
    values = np.cumsum(np.random.normal(0, 0.1, len(dates))) + 20
    
    # Create DataFrame
    data = pd.DataFrame({'Close': values}, index=dates)
    
    # Create and fit model
    model = ARIMAModel(order=(1, 1, 1))
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
    test_arima()