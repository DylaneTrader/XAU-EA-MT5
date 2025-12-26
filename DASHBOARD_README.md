# Streamlit Training Dashboard

## Overview
This Streamlit dashboard provides a user-friendly interface for training and evaluating the XAU-EA-MT5 Transformer model.

## Features
✅ **Data Loading**: Upload CSV/XLSX files with historical OHLCV data  
✅ **Technical Indicators**: Automatic calculation of RSI, MACD, Bollinger Bands, ATR  
✅ **Label Creation**: Generate trading signals based on future price movement  
✅ **Train/Test Split**: Configurable data splitting for validation  
✅ **Model Training**: Interactive training with real-time progress  
✅ **Evaluation**: Comprehensive metrics and visualizations  
✅ **Model Saving**: Save trained models to disk  

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Data Upload (Tab: Data Overview)
- Click "Browse files" to upload a CSV or XLSX file
- Required columns: `open`, `high`, `low`, `close`, `volume`
- Recommended: 5000+ bars for optimal training
- Click "Calculate Technical Indicators" to add RSI, MACD, Bollinger Bands, ATR

### 2. Configure Training (Sidebar)
- **Training Epochs**: Number of training iterations (default: 20)
- **Test Set Size**: Proportion of data for testing (default: 0.2)
- **Sequence Length**: Number of bars to look back (default: 60)
- **Forward Bars**: How many bars ahead to predict (default: 5)
- **Price Threshold**: Minimum price change for BUY/SELL signals (default: 0.001)
- **Hidden Dimension**: Model complexity (default: 128)
- **Transformer Layers**: Number of layers (default: 4)

### 3. Train Model (Tab: Training)
- Select features to use for training
- Click "Start Training" to begin
- Monitor progress in real-time
- View training loss and accuracy curves

### 4. Evaluate Model (Tab: Evaluation)
- View overall accuracy and confidence metrics
- Analyze confusion matrix
- Review classification report
- Compare prediction distributions

### 5. Save Model (Tab: Model Management)
- Click "Save Model to Disk" to save trained model
- Download model file for deployment
- View model information and architecture

## Data Format

Your CSV/XLSX file should have these columns:

| Column | Description | Required |
|--------|-------------|----------|
| open | Opening price | Yes |
| high | High price | Yes |
| low | Low price | Yes |
| close | Closing price | Yes |
| volume | Trading volume | Yes |
| time | Timestamp (optional) | No |

Example CSV:
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,2045.50,2046.80,2044.20,2045.90,150
2024-01-01 00:05:00,2045.90,2047.10,2045.30,2046.50,180
...
```

## Sample Data Template

A sample CSV template is provided in `sample_data_template.csv`. You can use this as a reference for formatting your own data.

## Trading Signals

The model predicts three classes:
- **0 (SELL)**: Price expected to decrease by more than threshold
- **1 (HOLD)**: Price expected to remain relatively stable
- **2 (BUY)**: Price expected to increase by more than threshold

## Model Output

When you save the model, it creates a `.pth` file containing:
- Model weights (state_dict)
- Feature list
- Scaler parameters (mean, scale)
- Sequence length
- Hidden dimension
- Training date and test accuracy

This file can be loaded by your EA for live trading.

## Troubleshooting

### "Missing required columns" error
Make sure your data file has columns named exactly: `open`, `high`, `low`, `close`, `volume` (lowercase)

### "Insufficient data for indicators" warning
Technical indicators need at least 50 bars. Upload more historical data.

### Model training is slow
- Reduce the number of epochs
- Use fewer training samples
- Reduce sequence length
- GPU acceleration is used if available

### Out of memory error
- Reduce batch size by using fewer samples
- Reduce sequence length
- Close other applications

## Tips for Better Results

1. **More data is better**: Use 5000+ bars for training
2. **Balance your labels**: Check the BUY/HOLD/SELL distribution
3. **Tune the threshold**: Adjust price threshold based on your symbol's volatility
4. **Feature selection**: Experiment with different combinations of features
5. **Hyperparameter tuning**: Try different epochs, hidden dimensions, and layers
6. **Avoid overfitting**: Monitor if training accuracy is much higher than test accuracy

## Next Steps

After training a satisfactory model:
1. Save the model using "Save Model to Disk"
2. Copy the `.pth` file to your MT5 EA directory
3. Configure your EA to use the trained model
4. Start live trading or backtesting

## Support

For issues or questions, please refer to the main project README.md
