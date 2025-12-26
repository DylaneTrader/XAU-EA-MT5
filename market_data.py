"""
Market data handler and technical indicators calculator
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Handler for MT5 market data and technical indicators"""
    
    def __init__(self, symbol, timeframe):
        """
        Initialize market data handler
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: MT5 timeframe constant
        """
        self.symbol = symbol
        self.timeframe = self._get_timeframe(timeframe)
        
    def _get_timeframe(self, timeframe_str):
        """Convert timeframe string to MT5 constant"""
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return timeframes.get(timeframe_str, mt5.TIMEFRAME_M15)
    
    def get_historical_data(self, num_bars):
        """
        Get historical OHLCV data from MT5
        
        Args:
            num_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, num_bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {self.symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Minimum bars needed: 50 for SMA, 26 for MACD, 14 for RSI
        min_bars_required = 50
        
        if df is None or len(df) < min_bars_required:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} < {min_bars_required}")
            return df
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        
        # Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Calculated indicators for {len(df)} bars")
        return df
    
    def prepare_features(self, df, feature_columns):
        """
        Prepare feature matrix for model
        
        Args:
            df: DataFrame with indicators
            feature_columns: List of column names to use as features
            
        Returns:
            Feature matrix
        """
        if df is None or len(df) == 0:
            return None
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            logger.error("No valid features found")
            return None
        
        features = df[available_features].values
        return features
    
    def get_current_price(self):
        """Get current bid/ask prices"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {self.symbol}")
            return None, None
        
        return tick.bid, tick.ask
    
    def get_market_features(self, num_bars, feature_columns):
        """
        Get market data and prepare features in one call
        
        Args:
            num_bars: Number of bars to retrieve
            feature_columns: List of feature column names
            
        Returns:
            Feature matrix ready for model
        """
        df = self.get_historical_data(num_bars)
        if df is None:
            return None
        
        df = self.calculate_indicators(df)
        if df is None:
            return None
        
        features = self.prepare_features(df, feature_columns)
        return features
