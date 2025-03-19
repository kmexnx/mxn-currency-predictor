#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the MXN Currency Predictor.

This script provides a command-line interface to predict currency exchange rates
based on historical MXN (Mexican Peso) data.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add the project directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collector import CurrencyDataCollector
from src.data.processor import DataProcessor
from src.models.arima import ARIMAModel
from src.models.prophet import ProphetModel
from src.models.lstm import LSTMModel
from src.models.linear import LinearRegressionModel
from src.visualization.plotter import plot_predictions, plot_comparison


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Predict currency exchange rates based on MXN (Mexican Peso)"
    )
    
    parser.add_argument(
        "--currency", 
        type=str, 
        default="USD",
        help="Currency code(s) to predict against MXN. For multiple currencies, use comma-separated list (e.g., USD,EUR,JPY)"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=30,
        help="Number of days to forecast"
    )
    
    parser.add_argument(
        "--models", 
        type=str, 
        default="prophet",
        help="Model(s) to use for prediction (prophet, arima, lstm, linear, all). For multiple models, use comma-separated list"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d"),
        help="Start date for historical data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for historical data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the prediction results and plots"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    
    return parser.parse_args()


def get_model_class(model_name):
    """
    Get the model class based on the model name.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        class: Model class
    """
    models = {
        "prophet": ProphetModel,
        "arima": ARIMAModel,
        "lstm": LSTMModel,
        "linear": LinearRegressionModel
    }
    
    return models.get(model_name.lower())


def main():
    """
    Main function to run the currency prediction.
    """
    args = parse_arguments()
    
    # Create output directory if saving results
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Parse currencies
    currencies = [c.strip().upper() for c in args.currency.split(",")]
    
    # Parse models
    if args.models.lower() == "all":
        models = ["prophet", "arima", "lstm", "linear"]
    else:
        models = [m.strip().lower() for m in args.models.split(",")]
    
    # Dictionary to store results
    results = {}
    
    # Process each currency
    for currency in currencies:
        print(f"\nProcessing {currency}/MXN exchange rate...")
        pair = f"{currency}/MXN"
        
        # Collect historical data
        collector = CurrencyDataCollector()
        historical_data = collector.get_historical_data(
            currency, "MXN", args.start_date, args.end_date
        )
        
        if historical_data is None or historical_data.empty:
            print(f"Error: Could not retrieve data for {pair}")
            continue
            
        print(f"Collected {len(historical_data)} data points for {pair}")
        
        # Process data
        processor = DataProcessor()
        processed_data = processor.process(historical_data)
        
        # Store currency results
        currency_results = {}
        
        # Run each model
        for model_name in models:
            model_class = get_model_class(model_name)
            
            if model_class is None:
                print(f"Warning: Unknown model '{model_name}', skipping...")
                continue
                
            try:
                print(f"Running {model_name.upper()} model for {pair}...")
                model = model_class()
                model.fit(processed_data)
                prediction = model.predict(args.days)
                
                metrics = model.evaluate()
                
                currency_results[model_name] = {
                    "prediction": prediction,
                    "metrics": metrics
                }
                
                print(f"  {model_name.upper()} metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
                
                # Plot individual model prediction
                fig = plot_predictions(processed_data, prediction, pair, model_name)
                
                if args.save:
                    fig_path = os.path.join(args.output_dir, f"{currency}_MXN_{model_name}.png")
                    fig.savefig(fig_path)
                    print(f"  Saved prediction plot to {fig_path}")
                    
                plt.close(fig)
                    
            except Exception as e:
                print(f"Error running {model_name} model for {pair}: {str(e)}")
                continue
        
        # Store results for this currency
        results[currency] = currency_results
        
        # Compare models if multiple models were used
        if len(currency_results) > 1:
            try:
                comparison_fig = plot_comparison(processed_data, currency_results, pair)
                
                if args.save and comparison_fig:
                    comp_path = os.path.join(args.output_dir, f"{currency}_MXN_comparison.png")
                    comparison_fig.savefig(comp_path)
                    print(f"Saved model comparison plot to {comp_path}")
                    
                if comparison_fig:
                    plt.close(comparison_fig)
            except Exception as e:
                print(f"Error creating comparison plot for {pair}: {str(e)}")
    
    # Save consolidated results if requested
    if args.save and results:
        try:
            # Create a summary DataFrame
            summary_rows = []
            
            for currency, currency_results in results.items():
                for model_name, model_data in currency_results.items():
                    metrics = model_data["metrics"]
                    
                    row = {
                        "Currency": f"{currency}/MXN",
                        "Model": model_name.upper(),
                        "RMSE": metrics["rmse"],
                        "MAE": metrics["mae"],
                        "R2": metrics.get("r2", float("nan")),
                        "Forecast Period": f"{args.days} days"
                    }
                    
                    summary_rows.append(row)
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_path = os.path.join(args.output_dir, "prediction_summary.csv")
                summary_df.to_csv(summary_path, index=False)
                print(f"\nSaved prediction summary to {summary_path}")
        except Exception as e:
            print(f"Error saving summary: {str(e)}")
    
    print("\nPrediction process completed!")


if __name__ == "__main__":
    main()