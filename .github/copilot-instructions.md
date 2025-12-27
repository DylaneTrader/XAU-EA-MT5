# XAU-EA-MT5 AI Coding Agent Instructions

## Project Overview
This is an **automated trading Expert Advisor (EA)** for MetaTrader 5 that uses **PyTorch Transformer neural networks** to predict XAUUSD (Gold) price movements. The system combines deep learning with technical analysis for real-time trading decisions.

## Architecture Flow
```
MT5 Market Data → MarketDataHandler → Feature Engineering → Transformer Model → TradeManager → MT5 Orders
                                       ↓
                                  [RSI, MACD, BB, ATR, MAs]
```

**Critical Components:**
- **main.py**: EA orchestration loop - checks positions, gets predictions, executes trades
- **transformer_model.py**: PyTorch Transformer (4 layers, 8 heads, 128 hidden dim) predicts BUY(2)/HOLD(1)/SELL(0)
- **market_data.py**: OHLCV retrieval + technical indicator calculation (RSI, MACD, Bollinger Bands, ATR)
- **trade_manager.py**: MT5 order execution, position management, SL/TP handling
- **config.py**: **Single source of truth** for all parameters (credentials, lot sizes, model hyperparameters, risk settings)

## Key Development Patterns

### 1. Configuration-Driven Design
**ALL parameters** live in [config.py](config.py). Never hardcode values - always reference `config.PARAMETER_NAME`.
```python
# ✅ Correct
self.lot_size = config.LOT_SIZE
# ❌ Wrong
self.lot_size = 0.01
```

### 2. MT5 Connection Lifecycle
MT5 must be initialized before ANY market operations and shutdown after. Always wrap in try/finally:
```python
if not mt5.initialize():
    logger.error("MT5 initialization failed")
    return False
try:
    # ... operations ...
finally:
    mt5.shutdown()
```

### 3. Feature Engineering Contract
The model expects features in **exact order** defined in `config.FEATURES`. MarketDataHandler must return numpy arrays with shape `(sequence_length, len(FEATURES))`. Missing or reordered features break predictions.

### 4. Signal Confidence Filtering
Model outputs require confidence thresholding (`config.PREDICTION_THRESHOLD`). Only signals with `confidence >= threshold` trigger trades. This prevents low-confidence noise from executing.

### 5. Magic Number Isolation
`config.MAGIC_NUMBER` ensures the EA only manages its own trades. TradeManager filters all position queries by magic number to avoid interfering with manual trades or other EAs.

## Critical Workflows

### Running the EA Live
```bash
python main.py
```
**Prerequisites**: MT5 terminal running, credentials in config.py, model trained (`transformer_ea_model.pth` exists)

### Training the Model
```bash
# Option 1: Script-based training
python train_model.py

# Option 2: Interactive dashboard (recommended)
streamlit run streamlit_dashboard.py
```
Training requires **5000+ historical bars** for meaningful patterns. Uses label creation based on forward price movement (`forward_bars` parameter).

### Testing Components
```bash
python test_ea.py
```
Runs isolated tests for MT5 connection, market data retrieval, model inference, and full pipeline.

### Deployment Workflow

**Demo Account Testing** (Always first step):
1. Configure demo credentials in `config.py` (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
2. Train model: `streamlit run streamlit_dashboard.py` or `python train_model.py`
3. Verify model file exists: `transformer_ea_model.pth`
4. Run on demo: `python main.py`
5. Monitor for 24-48 hours, verify trades execute correctly
6. Check logs for errors, verify SL/TP placement, monitor confidence levels

**Production Deployment** (Only after demo success):
1. Update `config.py` with live account credentials
2. Set conservative parameters: Lower `LOT_SIZE`, higher `PREDICTION_THRESHOLD` (0.65-0.70)
3. Enable MT5 algorithmic trading: Tools > Options > Expert Advisors > "Allow algorithmic trading"
4. Start EA: `python main.py`
5. Monitor continuously for first week
6. Use `Ctrl+C` for graceful shutdown (saves model state, closes positions optional)

**Critical Pre-Deployment Checklist**:
- [ ] Model trained on recent data (last 6-12 months minimum)
- [ ] Test accuracy >55% on validation set
- [ ] Demo account shows consistent behavior
- [ ] MT5 terminal running and logged in
- [ ] Symbol `XAUUSDm` visible (check `mt5.symbol_select()`)
- [ ] Sufficient margin for `MAX_TRADES` positions
- [ ] Stop-loss values appropriate for account size

### Backtesting Approach

**No Built-in Backtester** - The EA is designed for live trading. For backtesting:

**Option 1: Historical Simulation** (Manual):
1. Modify `main.py` to iterate through historical data instead of real-time
2. Comment out `time.sleep()` and MT5 order execution
3. Replace `market_data.get_market_features()` with historical DataFrame slicing
4. Log predicted signals and simulate fills at next bar's open
5. Calculate metrics: total trades, win rate, profit factor, max drawdown

**Option 2: MT5 Strategy Tester** (Requires MQL5 conversion):
- Current Python EA cannot run in MT5's built-in tester
- Would require rewriting logic in MQL5 (not recommended due to PyTorch dependency)

**Option 3: Custom Backtesting Script** (Recommended):
```python
# Load historical data
df = pd.read_csv('XAUUSDm_M5_20150101_20251226.csv')
# Calculate indicators via MarketDataHandler
# Loop through data, generate signals, track P&L
# No actual MT5 connection needed
```

**Performance Metrics to Track**:
- Win rate (target >50%)
- Profit factor (total wins / total losses, target >1.2)
- Maximum drawdown (target <20% of account)
- Average trade duration
- Sharpe ratio (if tracking daily returns)

## Model Architecture Specifics

**Input**: Sequences of shape `(batch, sequence_length, num_features)` where `sequence_length=60` and `num_features=11` (OHLCV + 6 indicators)

**Transformer Layers**: 
- Input projection: features → 128-dim hidden space
- 4 encoder layers with 8 attention heads each
- Classification head: hidden → 64 → 3 classes (BUY/HOLD/SELL)

**Output**: Softmax probabilities over 3 classes. Argmax gives signal, max probability gives confidence.

## Data Handling Conventions

### Symbol Naming
The system uses **XAUUSDm** (with 'm' suffix for micro lots on Exness). When adapting to other brokers, check symbol availability via `mt5.symbols_get()` and update `config.SYMBOL`.

### Timeframe Format
MT5 uses constants (`mt5.TIMEFRAME_M5`), but config stores as string (`"M5"`). MarketDataHandler converts strings to MT5 constants internally.

### Price/Pips Calculation
Stop-loss and take-profit use **pips** in config but must be converted to price using `symbol_info.point`:
```python
sl_price = entry_price - (stop_loss_pips * point * 10)  # For 5-digit brokers
```

## Dashboard Integration

The Streamlit dashboard ([streamlit_dashboard.py](streamlit_dashboard.py)) provides:
- Visual model training with real-time loss curves
- Confusion matrix and performance metrics
- Custom data upload (CSV/XLSX with required OHLCV columns)
- Model save/load management

**Default dataset**: `XAUUSDm_M5_20150101_20251226.csv` contains 1M+ bars (2015-2025) for training.

## Common Pitfalls

1. **Forgetting MT5 initialization**: Always call `mt5.initialize()` before any market operations
2. **Feature shape mismatches**: Model expects exactly `config.SEQUENCE_LENGTH` bars, not less
3. **Overtrading**: Check `max_trades` limit before opening positions
4. **Stale model**: Running untrained model produces random predictions - always train first
5. **Symbol not visible**: Call `mt5.symbol_select(symbol, True)` if symbol_info returns None
6. **Broker-specific symbols**: XAUUSDm works on Exness; other brokers may use XAUUSD, GOLD, or variants
7. **Timeframe string conversion**: Config uses "M5" but MT5 needs `mt5.TIMEFRAME_M5` constant
8. **Pips calculation errors**: 5-digit brokers require `point * 10` for pip conversion

## Dependencies
Core: `MetaTrader5`, `torch`, `numpy`, `pandas`, `scikit-learn`, `ta` (technical analysis)
Dashboard: `streamlit`, `matplotlib`, `seaborn`

## Version Requirements
- Python 3.8+
- PyTorch 2.0+ (for Transformer implementation)
- MetaTrader5 5.0.45+

## Next Steps for New Features
- **Adding indicators**: Update `market_data.py` calculation methods AND `config.FEATURES` list
- **Changing model architecture**: Modify `TransformerPricePredictor` in [transformer_model.py](transformer_model.py) AND retrain
- **Risk management changes**: Update trade logic in `main.py` execute_trading_logic() method
- **New symbols**: Update `config.SYMBOL` and test symbol availability in test_ea.py
