# MXN Currency Predictor

A Python tool for predicting currency exchange rates based on MXN (Mexican Peso) historical data.

## Features

- Historical currency data collection using Yahoo Finance API
- Multiple prediction models (ARIMA, LSTM, Prophet, Linear Regression)
- Customizable prediction timeframes
- Visualization tools for trend analysis
- Support for multiple currencies (USD, EUR, JPY, GBP, CAD, etc.)
- Performance metrics and model evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/kmexnx/mxn-currency-predictor.git
cd mxn-currency-predictor

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the main prediction tool with default settings (USD/MXN, 30 days forecast)
python src/main.py

# Predict specific currency against MXN with custom forecast days
python src/main.py --currency EUR --days 60

# Run with all available models
python src/main.py --currency USD --models all

# Get help on available options
python src/main.py --help
```

## Project Structure

```
mxn-currency-predictor/
├── src/
│   ├── __init__.py
│   ├── main.py                  # Main entry point
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py         # Data collection functionality
│   │   └── processor.py         # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima.py             # ARIMA model implementation
│   │   ├── lstm.py              # LSTM model implementation
│   │   ├── prophet.py           # Prophet model implementation
│   │   └── linear.py            # Linear regression model
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py           # Plotting functions
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks for experimentation
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## License

MIT