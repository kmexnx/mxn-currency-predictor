#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for currency prediction.

This module provides functions for plotting currency data and predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# Set Seaborn style
sns.set_style("whitegrid")


def plot_predictions(historical_data, predictions, pair, model_name):
    """
    Plot historical data and predictions.
    
    Args:
        historical_data (pandas.DataFrame): Historical data with DatetimeIndex and 'Close' column
        predictions (pandas.DataFrame): Prediction data with DatetimeIndex and 'Close' column
        pair (str): Currency pair name (e.g., "USD/MXN")
        model_name (str): Name of the model used for prediction
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical_data.index, historical_data['Close'], 
            label='Historical', color='blue', linewidth=2)
    
    # Plot predictions
    ax.plot(predictions.index, predictions['Close'], 
            label=f'{model_name.upper()} Prediction', 
            color='red', linewidth=2)
    
    # Plot confidence intervals if available
    if 'Lower' in predictions.columns and 'Upper' in predictions.columns:
        if not predictions['Lower'].isna().all() and not predictions['Upper'].isna().all():
            ax.fill_between(predictions.index, predictions['Lower'], predictions['Upper'],
                           color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Format the plot
    ax.set_title(f'{pair} Exchange Rate Prediction - {model_name.upper()} Model', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Exchange Rate', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add current date line
    current_date = historical_data.index[-1]
    ax.axvline(x=current_date, color='green', linestyle='--', linewidth=1.5,
              label='Current Date')
    
    # Adjust legend
    ax.legend(loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_comparison(historical_data, model_results, pair):
    """
    Plot comparison of different model predictions.
    
    Args:
        historical_data (pandas.DataFrame): Historical data with DatetimeIndex and 'Close' column
        model_results (dict): Dictionary with model predictions and metrics
        pair (str): Currency pair name (e.g., "USD/MXN")
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical_data.index, historical_data['Close'], 
            label='Historical', color='blue', linewidth=2)
    
    # Colors for different models
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Plot predictions for each model
    for i, (model_name, model_data) in enumerate(model_results.items()):
        predictions = model_data['prediction']
        color = colors[i % len(colors)]
        
        ax.plot(predictions.index, predictions['Close'], 
                label=f'{model_name.upper()} Prediction', 
                color=color, linewidth=2)
    
    # Format the plot
    ax.set_title(f'{pair} Exchange Rate Prediction - Model Comparison', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Exchange Rate', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add current date line
    current_date = historical_data.index[-1]
    ax.axvline(x=current_date, color='black', linestyle='--', linewidth=1.5,
              label='Current Date')
    
    # Create metrics table
    metrics_table = []
    headers = ['Model', 'RMSE', 'MAE']
    
    for model_name, model_data in model_results.items():
        metrics = model_data['metrics']
        metrics_table.append([model_name.upper(), 
                             f"{metrics['rmse']:.4f}", 
                             f"{metrics['mae']:.4f}"])
    
    if metrics_table:
        # Add table to the plot
        ax_table = plt.table(cellText=metrics_table,
                           colLabels=headers,
                           loc='bottom',
                           bbox=[0.0, -0.35, 1.0, 0.2],
                           cellLoc='center')
        
        ax_table.auto_set_font_size(False)
        ax_table.set_fontsize(9)
        ax_table.scale(1, 1.5)
        
        # Adjust figure size to accommodate table
        plt.subplots_adjust(bottom=0.25)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_historical_data(data, title):
    """
    Plot historical currency data.
    
    Args:
        data (pandas.DataFrame): Historical data with DatetimeIndex and 'Close' column
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.plot(data.index, data['Close'], label='Exchange Rate', color='blue', linewidth=2)
    
    # Add rolling averages
    if len(data) >= 30:
        ax.plot(data.index, data['Close'].rolling(window=7).mean(), 
                label='7-Day MA', color='red', linewidth=1.5, alpha=0.7)
        ax.plot(data.index, data['Close'].rolling(window=30).mean(), 
                label='30-Day MA', color='green', linewidth=1.5, alpha=0.7)
    
    # Calculate and plot trend line using polynomial fit
    if len(data) > 2:
        x = np.arange(len(data))
        z = np.polyfit(x, data['Close'].values, 1)
        p = np.poly1d(z)
        ax.plot(data.index, p(x), "r--", color='purple', linewidth=1.5, 
               label=f'Trend: {z[0]:.6f}x + {z[1]:.2f}')
    
    # Format the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Exchange Rate', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    # Format x-axis dates
    if len(data) > 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_volatility(data, window=30, title=None):
    """
    Plot exchange rate volatility.
    
    Args:
        data (pandas.DataFrame): Historical data with DatetimeIndex and 'Close' column
        window (int): Rolling window size for volatility calculation
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if len(data) < window:
        print(f"Warning: Not enough data for {window}-day volatility calculation")
        window = max(5, len(data) // 2)
    
    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()
    
    # Calculate rolling volatility (standard deviation of returns)
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot exchange rate
    ax1.plot(data.index, data['Close'], label='Exchange Rate', color='blue', linewidth=2)
    ax1.set_title(title or f'Exchange Rate and {window}-Day Volatility', fontsize=16)
    ax1.set_ylabel('Exchange Rate', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot volatility
    ax2.fill_between(volatility.index, volatility, color='red', alpha=0.5)
    ax2.plot(volatility.index, volatility, color='red', linewidth=1)
    ax2.set_title(f'{window}-Day Volatility', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Volatility', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in [ax1, ax2]:
        if len(data) > 60:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def test_plotter():
    """
    Test function for the plotting utilities.
    """
    # Create some sample data
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')
    np.random.seed(42)
    values = np.cumsum(np.random.normal(0, 0.1, len(dates))) + 20
    
    # Create DataFrame
    data = pd.DataFrame({'Close': values}, index=dates)
    
    # Create some sample predictions
    future_dates = pd.date_range(
        start=dates[-1] + timedelta(days=1),
        periods=30,
        freq='D'
    )
    future_values = values[-1] + np.cumsum(np.random.normal(0, 0.1, len(future_dates)))
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'Close': future_values,
        'Lower': future_values - 0.5,
        'Upper': future_values + 0.5
    }, index=future_dates)
    
    # Create model results dictionary
    model_results = {
        'arima': {
            'prediction': predictions,
            'metrics': {'rmse': 0.2, 'mae': 0.15}
        },
        'prophet': {
            'prediction': predictions.copy(),
            'metrics': {'rmse': 0.25, 'mae': 0.18}
        }
    }
    
    # Test plots
    fig1 = plot_predictions(data, predictions, "USD/MXN", "ARIMA")
    fig2 = plot_comparison(data, model_results, "USD/MXN")
    fig3 = plot_historical_data(data, "USD/MXN Historical Exchange Rate")
    fig4 = plot_volatility(data, window=10, title="USD/MXN Volatility Analysis")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    test_plotter()