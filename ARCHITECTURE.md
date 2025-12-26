# XAU-EA-MT5 Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MetaTrader 5 Terminal                   │
│                   (Real-time Market Data)                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ MT5 Python API
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      XAU-EA-MT5 System                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Main EA Loop (main.py)                   │  │
│  │  - Initialize system                                  │  │
│  │  - Monitor market continuously                        │  │
│  │  - Execute trading logic                              │  │
│  │  - Manage positions                                   │  │
│  └──────────┬───────────────────────────┬────────────────┘  │
│             │                           │                    │
│  ┌──────────▼──────────┐   ┌───────────▼────────────┐      │
│  │   Market Data       │   │   Trade Manager         │      │
│  │   (market_data.py)  │   │   (trade_manager.py)    │      │
│  │                     │   │                         │      │
│  │ • Get OHLCV data    │   │ • Open positions        │      │
│  │ • Calculate RSI     │   │ • Close positions       │      │
│  │ • Calculate MACD    │   │ • Manage SL/TP          │      │
│  │ • Bollinger Bands   │   │ • Check account status  │      │
│  │ • ATR, MAs          │   │                         │      │
│  │ • Feature prep      │   │                         │      │
│  └──────────┬──────────┘   └─────────────────────────┘      │
│             │                                                │
│  ┌──────────▼──────────────────────────────────────────┐   │
│  │       Transformer Model (transformer_model.py)       │   │
│  │                                                      │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │     Input Projection Layer                   │  │   │
│  │  │     (Features → Hidden Dimension)            │  │   │
│  │  └────────────────┬─────────────────────────────┘  │   │
│  │                   │                                 │   │
│  │  ┌────────────────▼─────────────────────────────┐  │   │
│  │  │   Transformer Encoder (4 layers, 8 heads)    │  │   │
│  │  │   - Self-attention mechanism                 │  │   │
│  │  │   - Feed-forward networks                    │  │   │
│  │  └────────────────┬─────────────────────────────┘  │   │
│  │                   │                                 │   │
│  │  ┌────────────────▼─────────────────────────────┐  │   │
│  │  │      Classification Head                     │  │   │
│  │  │      (BUY / HOLD / SELL)                     │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Configuration (config.py)                      │  │
│  │  - Trading parameters                                 │  │
│  │  - Risk management settings                           │  │
│  │  - Model hyperparameters                              │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. Market Data Collection
   MT5 → Historical OHLCV → market_data.py
   
2. Feature Engineering
   OHLCV → Technical Indicators → Feature Matrix
   (RSI, MACD, Bollinger Bands, ATR, MAs)
   
3. Model Prediction
   Feature Sequence → Transformer → [BUY/HOLD/SELL] + Confidence
   
4. Signal Filtering
   Prediction → Confidence Check → Valid Signal
   (Only if confidence > threshold)
   
5. Trade Execution
   Valid Signal → trade_manager.py → MT5 Order
   
6. Position Management
   Open Positions → Monitor SL/TP → Close if needed
```

## Component Details

### 1. Main EA (main.py)

**Purpose**: Orchestrate all components and manage the trading loop

**Key Functions**:
- `initialize()`: Setup MT5 connection and components
- `get_trading_signal()`: Get prediction from model
- `execute_trading_logic()`: Execute trades based on signals
- `run()`: Main event loop
- `shutdown()`: Cleanup and save state

**Data Flow**:
```
Initialize → Loop: [Get Data → Predict → Execute → Wait] → Shutdown
```

### 2. Market Data Handler (market_data.py)

**Purpose**: Interface with MT5 for market data and calculate indicators

**Key Classes**:
- `MarketDataHandler`: Main handler class

**Key Methods**:
- `get_historical_data(num_bars)`: Fetch OHLCV from MT5
- `calculate_indicators(df)`: Add technical indicators
- `prepare_features(df, features)`: Create feature matrix
- `get_market_features()`: Complete pipeline

**Technical Indicators**:
```
Price Data: Open, High, Low, Close, Volume
Momentum: RSI (14)
Trend: MACD (12,26,9), SMA (20,50), EMA (12,26)
Volatility: Bollinger Bands (20,2), ATR (14)
```

### 3. Trade Manager (trade_manager.py)

**Purpose**: Execute and manage trading orders

**Key Classes**:
- `TradeManager`: Order execution manager

**Key Methods**:
- `initialize()`: Setup symbol information
- `open_buy_position()`: Execute buy order
- `open_sell_position()`: Execute sell order
- `close_position()`: Close specific position
- `get_open_positions()`: Query current positions
- `get_account_info()`: Get account status

**Order Parameters**:
```
Symbol: XAUUSD
Volume: Configurable lot size
Type: Market order (TRADE_ACTION_DEAL)
SL/TP: Calculated based on pips
Magic Number: Unique EA identifier
```

### 4. Transformer Model (transformer_model.py)

**Purpose**: Neural network for price prediction

**Key Classes**:
- `TransformerPricePredictor`: Neural network model
- `ModelManager`: Model lifecycle management

**Architecture**:
```
Input: (batch, sequence_length, input_features)
       ↓
Input Projection: Linear(input_features → hidden_dim)
       ↓
Transformer Encoder: 4 layers × 8 attention heads
  - Multi-head self-attention
  - Position-wise feedforward
  - Layer normalization
  - Residual connections
       ↓
Take last time step: (batch, hidden_dim)
       ↓
Classification Head:
  - Linear(hidden_dim → hidden_dim/2)
  - ReLU + Dropout(0.1)
  - Linear(hidden_dim/2 → 3)
  - Softmax
       ↓
Output: [P(SELL), P(HOLD), P(BUY)]
```

**Model Parameters**:
- Input dimension: Number of features (11 default)
- Hidden dimension: 128
- Transformer layers: 4
- Attention heads: 8
- Dropout: 0.1
- Output classes: 3 (SELL, HOLD, BUY)

### 5. Configuration (config.py)

**Purpose**: Centralized configuration management

**Parameter Categories**:

1. **MT5 Connection**:
   - Login credentials (optional)
   - Server information

2. **Trading Parameters**:
   - Symbol, timeframe
   - Lot size, magic number
   - Max concurrent trades

3. **Risk Management**:
   - Stop loss/take profit (pips)
   - Risk percentage

4. **Model Parameters**:
   - Sequence length
   - Prediction threshold
   - Model architecture

5. **Feature Configuration**:
   - Technical indicators to use
   - Feature list

## Trading Logic Flow

```
START
  │
  ├─→ Check open positions
  │   └─→ If >= MAX_TRADES → Skip
  │
  ├─→ Get market features (last N bars)
  │   └─→ Features = [OHLCV + Indicators]
  │
  ├─→ Prepare sequence (SEQUENCE_LENGTH bars)
  │
  ├─→ Model prediction
  │   └─→ signal, confidence = model.predict(sequence)
  │
  ├─→ Check confidence
  │   └─→ If confidence < THRESHOLD → Skip
  │
  ├─→ Execute trade
  │   ├─→ If signal == BUY (2)
  │   │   └─→ Open long position
  │   ├─→ If signal == SELL (0)
  │   │   └─→ Open short position
  │   └─→ If signal == HOLD (1)
  │       └─→ No action
  │
  └─→ Wait PREDICTION_INTERVAL seconds
      └─→ REPEAT
```

## Risk Management

### Position Sizing
- Fixed lot size (configurable)
- Can be extended to use risk percentage

### Stop Loss / Take Profit
```
For BUY:
  SL = Entry Price - (STOP_LOSS_PIPS × Point × 10)
  TP = Entry Price + (TAKE_PROFIT_PIPS × Point × 10)

For SELL:
  SL = Entry Price + (STOP_LOSS_PIPS × Point × 10)
  TP = Entry Price - (TAKE_PROFIT_PIPS × Point × 10)
```

### Trade Filtering
- Confidence threshold prevents low-quality signals
- Maximum trades limit prevents overexposure
- Magic number isolates EA trades

## Model Training

### Label Creation (train_model.py)
```
For each bar:
  future_price = close[i + forward_bars]
  price_change = (future_price - current_price) / current_price
  
  If price_change > threshold:
    label = BUY (2)
  Else if price_change < -threshold:
    label = SELL (0)
  Else:
    label = HOLD (1)
```

### Training Process
```
1. Get historical data (5000+ bars)
2. Calculate technical indicators
3. Create labels based on future price
4. Split into train/test sets
5. Train model (20 epochs default)
6. Evaluate on test set
7. Save model to disk
```

### Model Persistence
- Saved as: `transformer_ea_model.pth`
- Contains: model state, scaler, training flag
- Auto-loaded on EA startup if exists

## Testing Strategy

### Unit Tests (test_ea.py)

1. **MT5 Connection Test**:
   - Verify MT5 initialization
   - Check terminal info

2. **Market Data Test**:
   - Fetch historical data
   - Calculate indicators
   - Prepare features

3. **Model Test**:
   - Initialize model
   - Test prediction
   - Test training loop

4. **Pipeline Test**:
   - End-to-end data → prediction
   - Real market data
   - Signal generation

## Performance Considerations

### Optimization Strategies

1. **Prediction Speed**:
   - Smaller models = faster inference
   - Reduce sequence length
   - Use CPU for small models

2. **Data Efficiency**:
   - Cache indicator calculations
   - Reuse historical data
   - Update incrementally

3. **Resource Usage**:
   - Model runs on CPU by default
   - GPU optional for training
   - Minimal memory footprint

### Monitoring

**Log Information**:
- Signal predictions and confidence
- Trade executions
- Position status
- Account metrics
- Errors and warnings

**Performance Metrics**:
- Win rate
- Profit factor
- Maximum drawdown
- Average trade duration

## Security Considerations

1. **Credentials**:
   - Store in .env file
   - Never commit to git
   - Use environment variables

2. **Risk Limits**:
   - Stop loss always set
   - Maximum position limit
   - Account equity checks

3. **Error Handling**:
   - Try-catch on all trades
   - Graceful shutdown
   - State persistence

## Future Enhancements

Possible improvements:

1. **Model Enhancements**:
   - Multi-timeframe analysis
   - Ensemble methods
   - Online learning

2. **Feature Engineering**:
   - Order book data
   - Sentiment analysis
   - Market microstructure

3. **Trading Strategy**:
   - Partial position closing
   - Trailing stops
   - Dynamic position sizing

4. **System Features**:
   - Web dashboard
   - Performance analytics
   - Backtesting framework
   - Telegram notifications

## Troubleshooting Guide

### Common Issues

**Issue**: MT5 connection failed
- **Solution**: Ensure MT5 terminal is running
- Check algorithmic trading is enabled

**Issue**: Insufficient data
- **Solution**: Wait for more bars to accumulate
- Reduce SEQUENCE_LENGTH

**Issue**: No trades executed
- **Solution**: Lower PREDICTION_THRESHOLD
- Check if MAX_TRADES allows new positions

**Issue**: Poor prediction quality
- **Solution**: Train model on historical data
- Add more features
- Increase sequence length

## References

- MetaTrader 5 Python API: https://www.mql5.com/en/docs/python_metatrader5
- PyTorch: https://pytorch.org/docs/stable/index.html
- Transformers: https://huggingface.co/docs/transformers/
- Technical Analysis Library: https://technical-analysis-library-in-python.readthedocs.io/

---

For more information, see README.md or contact the maintainers.
