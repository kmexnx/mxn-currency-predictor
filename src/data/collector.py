#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Currency data collection functionality.

This module provides tools to collect historical currency exchange rate data
from various sources such as Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time


class CurrencyDataCollector:
    """
Class for collecting currency exchange rate data.
    """
    
    def __init__(self):
        """
Initialize the CurrencyDataCollector.
        """
        self.source = "yahoo"  # Default source
        
    def get_historical_data(self, base_currency, quote_currency="MXN", 
                           start_date=None, end_date=None, source=None):
        """
        Get historical exchange rate data for a currency pair.
        
        Args:
            base_currency (str): Base currency code (e.g., "USD")
            quote_currency (str): Quote currency code (e.g., "MXN")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            source (str, optional): Data source. Default is set in __init__
            
        Returns:
            pandas.DataFrame: Historical data with DatetimeIndex and 'Close' column
        """
        source = source or self.source
        
        # Default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            if source.lower() == "yahoo":
                return self._get_yahoo_data(base_currency, quote_currency, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            print(f"Error retrieving data: {str(e)}")
            return None
    
    def _get_yahoo_data(self, base_currency, quote_currency, start_date, end_date):
        """
        Get historical data from Yahoo Finance.
        
        Args:
            base_currency (str): Base currency code
            quote_currency (str): Quote currency code
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Historical data
        """
        # Yahoo Finance symbol format
        symbol = f"{base_currency}{quote_currency}=X"
        
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data available for {symbol}")
            return None
        
        # Keep only the 'Close' price
        if 'Close' in data.columns:
            return data[['Close']]
        elif 'Adj Close' in data.columns:
            # Rename 'Adj Close' to 'Close' for consistency
            data = data[['Adj Close']]
            data.columns = ['Close']
            return data
        else:
            return None
    
    def get_latest_rate(self, base_currency, quote_currency="MXN"):
        """
        Get the latest exchange rate for a currency pair.
        
        Args:
            base_currency (str): Base currency code
            quote_currency (str): Quote currency code
            
        Returns:
            float: Latest exchange rate
        """
        try:
            # Get data for the last 7 days to ensure we have the latest rate
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            data = self.get_historical_data(base_currency, quote_currency, 
                                          start_date, end_date)
            
            if data is not None and not data.empty:
                # Return the most recent rate
                return data['Close'].iloc[-1]
            else:
                return None
        except Exception as e:
            print(f"Error retrieving latest rate: {str(e)}")
            return None


def test_collector():
    """
    Test function for the CurrencyDataCollector.
    """
    collector = CurrencyDataCollector()
    
    # Test getting historical data
    print("Testing historical data collection...")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    data = collector.get_historical_data("USD", "MXN", start_date, end_date)
    
    if data is not None:
        print(f"Retrieved {len(data)} data points for USD/MXN")
        print(data.head())
    else:
        print("Failed to retrieve data")
    
    # Test getting latest rate
    print("\nTesting latest rate retrieval...")
    latest = collector.get_latest_rate("USD", "MXN")
    
    if latest is not None:
        print(f"Latest USD/MXN rate: {latest:.4f}")
    else:
        print("Failed to retrieve latest rate")


if __name__ == "__main__":
    test_collector()