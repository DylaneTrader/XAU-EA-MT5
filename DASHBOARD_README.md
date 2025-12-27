# Streamlit Training Dashboard

## Overview
This Streamlit dashboard provides a user-friendly interface for training and evaluating the XAU-EA-MT5 Transformer model.

## Features
✅ **Data Loading**: Use default XAUUSD data or upload custom CSV/XLSX files  
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
- **Default Data**: Select "Use Default Data" to use the included XAUUSD 5-minute data from 2015-2025
- **Custom Data**: Select "Upload Custom File" to browse and upload your own CSV or XLSX file
- Required columns: `open`, `high`, `low`, `close`, `volume` (or `tick_volume`)
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
| volume | Trading volume | Yes* |
| tick_volume | Tick volume (alternative to volume) | Yes* |
| time | Timestamp (optional) | No |

*Either `volume` or `tick_volume` is required

Example CSV:
```csv
time,open,high,low,close,tick_volume,spread,real_volume
2024-07-24 12:10:00,2417.809,2418.479,2417.186,2417.62,558,179,0
2024-07-24 12:15:00,2417.62,2418.10,2417.30,2417.85,180,175,0
...
```

## Default Data

The dashboard includes `XAUUSDm_M5_20150101_20251226.csv` - XAUUSD 5-minute historical data from 2015 to December 2025. This data was downloaded directly from MT5 and contains over 1 million bars for comprehensive model training.

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
