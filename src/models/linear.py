#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear Regression model for currency prediction.

This module implements a Linear Regression model for predicting currency
exchange rates.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta


class LinearRegressionModel:
    """
    Linear Regression model for currency prediction.
    """
    
    def __init__(self):
        """
        Initialize the Linear Regression model.
        """
        self.model = LinearRegression()
        self.data = None
        self.features = None
        self.test_size = 30  # Default test size
        
    def _create_features(self, data):
        """
        Create features for the linear regression model.
        
        Args:
            data (pandas.DataFrame): Input data
            
        Returns:
            pandas.DataFrame: Feature DataFrame
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Create time-based features
        df['day_of_week'] = df.index.dayofweek  # 0 is Monday, 6 is Sunday
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['quarter'] = df.index.quarter
        
        # Create lag features
        for i in range(1, 8):  # Last 7 days
            df[f'lag_{i}'] = df['Close'].shift(i)
        
        # Create rolling window features
        df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
        df['rolling_mean_14'] = df['Close'].rolling(window=14).mean()
        df['rolling_std_7'] = df['Close'].rolling(window=7).std()
        
        # Create momentum features
        df['momentum_1'] = df['Close'].pct_change(periods=1)
        df['momentum_3'] = df['Close'].pct_change(periods=3)
        df['momentum_7'] = df['Close'].pct_change(periods=7)
        
        # Fill NaN values that were created
        df = df.fillna(method='bfill')
        
        # Define features to use
        feature_columns = [
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
            'momentum_1', 'momentum_3', 'momentum_7',
            'day_of_week', 'month', 'day', 'quarter'
        ]
        
        return df[feature_columns]
    
    def fit(self, data, test_size=None):
        """
        Fit the Linear Regression model to the data.
        
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
        
        # Create features
        self.features = self._create_features(self.data)
        
        # Split data into train and test sets
        self.X_train = self.features.iloc[:-self.test_size]
        self.y_train = self.data['Close'].iloc[:-self.test_size]
        
        self.X_test = self.features.iloc[-self.test_size:]
        self.y_test = self.data['Close'].iloc[-self.test_size:]
        
        # Fit the model
        try:
            self.model.fit(self.X_train, self.y_train)
            return self
        except Exception as e:
            raise Exception(f"Failed to fit Linear Regression model: {str(e)}")
    
    def predict(self, steps):
        """
        Make predictions for future time steps.
        
        Args:
            steps (int): Number of steps to predict
            
        Returns:
            pandas.DataFrame: DataFrame with DatetimeIndex and predictions
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the last date in the data
        last_date = self.data.index[-1]
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Create an empty DataFrame for predictions
        predictions = pd.DataFrame(index=future_dates)
        predictions['Close'] = 0  # Will be updated
        
        # Copy the last rows of data to use for initial lag values
        future_data = self.data.copy().iloc[-max(7, steps):]
        
        # Initial feature values
        for i in range(steps):
            # Current prediction date
            current_date = future_dates[i]
            
            # Append a row for the current date
            future_data.loc[current_date] = np.nan
            future_data.loc[current_date, 'Close'] = 0  # Temporary value
            
            # Create features for this date
            temp_features = self._create_features(future_data)
            current_features = temp_features.loc[current_date:current_date]
            
            # Predict
            prediction = self.model.predict(current_features)[0]
            
            # Update the prediction value
            predictions.loc[current_date, 'Close'] = prediction
            future_data.loc[current_date, 'Close'] = prediction
        
        # Add placeholder columns for confidence intervals
        predictions['Lower'] = np.nan
        predictions['Upper'] = np.nan
        
        return predictions
    
    def evaluate(self):
        """
        Evaluate the model performance on the test set.
        
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Make predictions on the test set
        predictions = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


def test_linear():
    """
    Test function for the Linear Regression model.
    """
    # Create some sample data
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')
    np.random.seed(42)
    values = np.cumsum(np.random.normal(0, 0.1, len(dates))) + 20
    
    # Create DataFrame
    data = pd.DataFrame({'Close': values}, index=dates)
    
    # Create and fit model
    model = LinearRegressionModel()
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
    test_linear()