#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM model for currency prediction.

This module implements a Long Short-Term Memory (LSTM) neural network model
for predicting currency exchange rates.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=Warning)


class LSTMModel:
    """
    LSTM model for currency prediction.
    """
    
    def __init__(self, seq_length=60, batch_size=32, epochs=100, early_stopping=True):
        """
        Initialize the LSTM model.
        
        Args:
            seq_length (int): Sequence length for LSTM input
            batch_size (int): Batch size for training
            epochs (int): Number of epochs for training
            early_stopping (bool): Whether to use early stopping
        """
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.model = None
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.test_size = 30  # Default test size
        
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (seq_length, n_features)
            
        Returns:
            tensorflow.keras.models.Sequential: LSTM model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM input.
        
        Args:
            data (numpy.ndarray): Input data array
            
        Returns:
            tuple: (X, y) where X is the sequences and y is the target values
        """
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
            
        return np.array(X), np.array(y)
    
    def fit(self, data, test_size=None):
        """
        Fit the LSTM model to the data.
        
        Args:
            data (pandas.DataFrame): Data with DatetimeIndex and 'Close' column
            test_size (int, optional): Number of data points to use for testing
            
        Returns:
            self: The fitted model instance
        """
        if data is None or data.empty:
            raise ValueError("No data provided")
            
        if len(data) <= self.seq_length:
            raise ValueError(f"Not enough data points. Need more than {self.seq_length}")
            
        self.data = data.copy()
        
        if test_size is not None:
            self.test_size = test_size
        
        # Scale the data
        values = self.data['Close'].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(values)
        
        # Split data into train and test sets
        train_data = self.scaled_data[:-self.test_size]
        test_data = self.scaled_data[-self.test_size-self.seq_length:]
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Reshape input for LSTM [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Build model
        self.model = self.build_model((self.seq_length, 1))
        
        # Callbacks
        callbacks = []
        if self.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ))
        
        # Fit model
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            self.history = history.history
            return self
        except Exception as e:
            raise Exception(f"Failed to fit LSTM model: {str(e)}")
    
    def predict(self, steps):
        """
        Make predictions for future time steps.
        
        Args:
            steps (int): Number of steps to predict
            
        Returns:
            pandas.DataFrame: DataFrame with DatetimeIndex and predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the last sequence from the data
        last_sequence = self.scaled_data[-self.seq_length:].reshape(1, self.seq_length, 1)
        
        # List to store predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        # Predict step by step
        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(current_sequence)[0][0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :], [[pred]], axis=1)
        
        # Convert predictions back to original scale
        predicted_values = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create dates for predictions
        last_date = self.data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Create prediction DataFrame
        predictions_df = pd.DataFrame({
            'Close': predicted_values.flatten(),
            'Lower': np.nan,  # LSTM doesn't provide confidence intervals by default
            'Upper': np.nan
        }, index=future_dates)
        
        return predictions_df
    
    def evaluate(self):
        """
        Evaluate the model performance on the test set.
        
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Predict on test data
        predictions = self.model.predict(self.X_test)
        
        # Convert back to original scale
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }


def test_lstm():
    """
    Test function for the LSTM model.
    """
    # Suppress TensorFlow warnings during test
    tf.get_logger().setLevel('ERROR')
    
    # Create some sample data (need more data for LSTM)
    dates = pd.date_range(start='2022-01-01', end='2023-03-31')
    np.random.seed(42)
    values = np.cumsum(np.random.normal(0, 0.1, len(dates))) + 20
    
    # Create DataFrame
    data = pd.DataFrame({'Close': values}, index=dates)
    
    # Create and fit model with fewer epochs for testing
    model = LSTMModel(seq_length=30, epochs=5)
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
    test_lstm()