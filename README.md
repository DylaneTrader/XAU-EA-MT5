# XAU-EA-MT5: Transformer-Based Expert Advisor

An advanced Expert Advisor (EA) for MetaTrader 5 that uses Transformer neural networks to trade XAUUSD (Gold). This EA leverages cutting-edge deep learning technology to analyze market patterns and make intelligent trading decisions.

## Features

- 🤖 **Transformer Neural Network**: Uses state-of-the-art Transformer architecture for price prediction
- 📊 **Technical Indicators**: Incorporates RSI, MACD, Bollinger Bands, ATR, and moving averages
- 💹 **Automated Trading**: Fully automated buy/sell signal generation and execution
- 🎯 **Risk Management**: Configurable stop-loss, take-profit, and position sizing
- 📈 **Real-time Analysis**: Continuous market monitoring and prediction
- 🔧 **Highly Configurable**: Easy customization through config file

## Architecture

The EA consists of several key components:

1. **transformer_model.py**: Transformer neural network implementation for price prediction
2. **market_data.py**: Market data retrieval and technical indicator calculation
3. **trade_manager.py**: Order execution and position management
4. **main.py**: Main EA logic and execution loop
5. **config.py**: Configuration parameters
6. **test_ea.py**: Testing and validation utilities

## Requirements

- MetaTrader 5 terminal
- Python 3.8 or higher
- Required Python packages (see requirements.txt)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DylaneTrader/XAU-EA-MT5.git
   cd XAU-EA-MT5
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure MetaTrader 5**:
   - Ensure MT5 terminal is installed and running
   - Enable algorithmic trading in MT5 (Tools > Options > Expert Advisors)
   - Make sure Python API is allowed

4. **Configure the EA**:
   Edit `config.py` to set your parameters:
   ```python
   # MT5 Connection (optional for demo accounts)
   MT5_LOGIN = None  # Your MT5 login
   MT5_PASSWORD = None  # Your MT5 password
   MT5_SERVER = None  # Your MT5 server
   
   # Trading Parameters
   SYMBOL = "XAUUSD"
   TIMEFRAME = "M15"
   LOT_SIZE = 0.01
   STOP_LOSS_PIPS = 50
   TAKE_PROFIT_PIPS = 100
   ```

## Usage

### Running the EA

To start the Expert Advisor:

```bash
python main.py
```

The EA will:
1. Connect to MetaTrader 5
2. Initialize the Transformer model
3. Start monitoring the market
4. Execute trades based on model predictions

### Testing the EA

Before running live, test the components:

```bash
python test_ea.py
```

This will run a comprehensive test suite including:
- MT5 connection test
- Market data retrieval test
- Transformer model test
- Full pipeline test

### Stopping the EA

Press `Ctrl+C` to gracefully stop the EA. It will:
- Close the main loop
- Save the model state
- Display final statistics
- Shutdown MT5 connection

## Configuration Options

### Trading Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SYMBOL` | Trading symbol | "XAUUSD" |
| `TIMEFRAME` | Candle timeframe | "M15" |
| `LOT_SIZE` | Position size | 0.01 |
| `MAGIC_NUMBER` | Unique EA identifier | 234000 |
| `MAX_TRADES` | Maximum concurrent positions | 1 |

### Risk Management

| Parameter | Description | Default |
|-----------|-------------|---------|
| `STOP_LOSS_PIPS` | Stop loss distance | 50 |
| `TAKE_PROFIT_PIPS` | Take profit distance | 100 |
| `RISK_PERCENT` | Risk per trade | 1.0% |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SEQUENCE_LENGTH` | Lookback period (candles) | 60 |
| `PREDICTION_THRESHOLD` | Minimum confidence for trade | 0.6 |
| `MODEL_HIDDEN_DIM` | Transformer hidden dimension | 128 |
| `MODEL_NUM_LAYERS` | Number of transformer layers | 4 |
| `MODEL_NUM_HEADS` | Number of attention heads | 8 |
| `PREDICTION_INTERVAL` | Time between predictions (sec) | 60 |

## How It Works

1. **Data Collection**: The EA retrieves historical OHLCV data from MT5 for the specified symbol and timeframe

2. **Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands, ATR, Moving Averages) are calculated and combined with price data

3. **Model Prediction**: The Transformer model analyzes the feature sequence and outputs:
   - Signal: BUY (2), HOLD (1), or SELL (0)
   - Confidence: Probability score (0.0 to 1.0)

4. **Trade Execution**: If confidence exceeds the threshold:
   - BUY signal → Open long position
   - SELL signal → Open short position
   - HOLD signal → No action

5. **Risk Management**: Each trade includes:
   - Calculated stop-loss level
   - Calculated take-profit level
   - Position size based on risk parameters

## Transformer Model

The EA uses a custom Transformer architecture built with PyTorch specifically adapted for time series forecasting:

- **Input Layer**: Projects market features to hidden dimension (128)
- **Transformer Encoder**: 4 layers with 8 attention heads, multi-head self-attention
- **Classification Head**: Outputs BUY/SELL/HOLD probabilities with softmax

The model architecture uses PyTorch's native transformer layers (not pre-trained language models), making it:
- Lightweight and fast
- Customizable for financial data
- Easy to train on historical market data

The model can be:
- Used untrained (random initialization) for initial testing
- Trained on historical data using the provided training script
- Loaded from saved checkpoint (automatically saved/loaded)

## Monitoring

The EA provides real-time logging of:
- Current signal and confidence
- Open positions and profit/loss
- Account balance and equity
- Trade execution results
- Errors and warnings

Example output:
```
2024-12-26 14:30:15 - Signal: BUY, Confidence: 0.7234
2024-12-26 14:30:16 - Opening BUY position (confidence: 0.7234)
2024-12-26 14:30:17 - Buy order successful: 12345678 at 2045.32
```

## Safety Considerations

⚠️ **Important Notes**:

1. **Demo Account First**: Always test on a demo account before using real money
2. **Market Hours**: Ensure you trade during appropriate market hours for XAUUSD
3. **Spread & Slippage**: Account for broker spreads and potential slippage
4. **Model Training**: The default model is untrained. Consider training on historical data
5. **Risk Management**: Never risk more than you can afford to lose
6. **Monitoring**: Regularly monitor the EA's performance and behavior

## Customization

### Adding Custom Indicators

Edit `market_data.py` to add new indicators:

```python
def calculate_indicators(self, df):
    # Add your custom indicator
    df['custom_indicator'] = your_calculation(df)
    return df
```

Then add to `config.py`:
```python
FEATURES = [..., 'custom_indicator']
```

### Adjusting Model Architecture

Edit `transformer_model.py` to modify:
- Number of transformer layers
- Attention heads
- Hidden dimensions
- Output classes

### Custom Trading Logic

Modify `execute_trading_logic()` in `main.py` to implement:
- Custom entry/exit rules
- Position management strategies
- Multiple timeframe analysis

## Troubleshooting

**MT5 Connection Failed**:
- Ensure MT5 terminal is running
- Check if algorithmic trading is enabled
- Verify login credentials (if using live account)

**Insufficient Data**:
- Wait for more bars to accumulate
- Reduce `SEQUENCE_LENGTH` in config
- Check if symbol is available in MT5

**No Trades Executed**:
- Check if `PREDICTION_THRESHOLD` is too high
- Verify `MAX_TRADES` allows new positions
- Ensure sufficient account margin

**Import Errors**:
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

## Performance Optimization

1. **Reduce Prediction Interval**: Decrease `PREDICTION_INTERVAL` for faster reactions (uses more CPU)
2. **Adjust Sequence Length**: Shorter sequences = faster predictions but less context
3. **Model Complexity**: Smaller models (fewer layers) = faster inference
4. **Feature Selection**: Use only most relevant indicators

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New features
- Performance improvements
- Documentation updates

## License

This project is open source and available under the MIT License.

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments carries risk. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Always test thoroughly on demo accounts and never trade with money you cannot afford to lose.

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the test script for examples

---

**Happy Trading! 📈💰**
