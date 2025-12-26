"""
Configuration file for XAU-EA-MT5 Expert Advisor
"""

# MT5 Connection Settings
MT5_LOGIN = 297581462  # Set your MT5 login
MT5_PASSWORD = "#Trader001"  # Set your MT5 password
MT5_SERVER = "Exness-MT5Trial9"  # Set your MT5 server

# Trading Parameters
SYMBOL = "XAUUSD"
TIMEFRAME = "M5"  # 5-minute timeframe
LOT_SIZE = 0.01  # Trading lot size
MAGIC_NUMBER = 234000  # Unique identifier for EA trades

# Risk Management
STOP_LOSS_PIPS = 500  # Stop loss in pips
TAKE_PROFIT_PIPS = 1000  # Take profit in pips
MAX_TRADES = 2  # Maximum concurrent trades
RISK_PERCENT = 5.0  # Risk percentage per trade

# Model Parameters
SEQUENCE_LENGTH = 60  # Number of candles to look back
PREDICTION_THRESHOLD = 0.6  # Confidence threshold for trade signals
MODEL_HIDDEN_DIM = 128  # Transformer hidden dimension
MODEL_NUM_LAYERS = 4  # Number of transformer encoder layers
MODEL_NUM_HEADS = 8  # Number of attention heads

# Technical Indicators
USE_RSI = True
USE_MACD = True
USE_BOLLINGER = True
USE_ATR = True

# Data Parameters
BARS_TO_LOAD = 1000  # Number of historical bars to load
PREDICTION_INTERVAL = 60  # Seconds between predictions
MODEL_PATH = 'transformer_ea_model.pth'  # Path to save/load model

# Feature Engineering
FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr'
]
