"""
Example script to demonstrate and test the Transformer EA functionality
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from market_data import MarketDataHandler
from transformer_model import ModelManager
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mt5_connection():
    """Test MT5 connection"""
    logger.info("Testing MT5 connection...")
    
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    
    logger.info(f"MT5 Version: {mt5.version()}")
    logger.info(f"Terminal Info: {mt5.terminal_info()}")
    
    mt5.shutdown()
    return True


def test_market_data():
    """Test market data retrieval and indicator calculation"""
    logger.info("\nTesting market data handler...")
    
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    
    handler = MarketDataHandler(config.SYMBOL, config.TIMEFRAME)
    
    # Get historical data
    df = handler.get_historical_data(100)
    
    if df is not None:
        logger.info(f"Retrieved {len(df)} bars")
        logger.info(f"Latest close price: {df['close'].iloc[-1]}")
        
        # Calculate indicators
        df_with_indicators = handler.calculate_indicators(df)
        
        if df_with_indicators is not None:
            logger.info(f"Calculated indicators for {len(df_with_indicators)} bars")
            logger.info("\nLatest indicator values:")
            logger.info(f"RSI: {df_with_indicators['rsi'].iloc[-1]:.2f}")
            logger.info(f"MACD: {df_with_indicators['macd'].iloc[-1]:.4f}")
            logger.info(f"ATR: {df_with_indicators['atr'].iloc[-1]:.4f}")
            
            # Get features
            features = handler.prepare_features(df_with_indicators, config.FEATURES)
            if features is not None:
                logger.info(f"\nFeature matrix shape: {features.shape}")
                logger.info("Market data test PASSED")
            else:
                logger.error("Failed to prepare features")
        else:
            logger.error("Failed to calculate indicators")
    else:
        logger.error("Failed to retrieve market data")
    
    mt5.shutdown()
    return True


def test_transformer_model():
    """Test transformer model initialization and prediction"""
    logger.info("\nTesting Transformer model...")
    
    num_features = len(config.FEATURES)
    model_manager = ModelManager(
        config.MODEL_NAME, 
        num_features, 
        config.SEQUENCE_LENGTH
    )
    
    # Create dummy data for testing
    dummy_data = np.random.randn(config.SEQUENCE_LENGTH, num_features)
    
    logger.info(f"Input shape: {dummy_data.shape}")
    
    # Test prediction (without training)
    signal, confidence = model_manager.predict_signal(dummy_data)
    
    signals = {0: "SELL", 1: "HOLD", 2: "BUY"}
    logger.info(f"Prediction: {signals[signal]} (confidence: {confidence:.4f})")
    
    # Test simple training with dummy data
    logger.info("\nTesting model training with dummy data...")
    X = np.random.randn(50, config.SEQUENCE_LENGTH, num_features)
    y = np.random.randint(0, 3, size=50)
    
    model_manager.train_simple(X, y, epochs=5)
    
    # Test prediction after training
    signal, confidence = model_manager.predict_signal(dummy_data)
    logger.info(f"Post-training prediction: {signals[signal]} (confidence: {confidence:.4f})")
    
    logger.info("Transformer model test PASSED")
    return True


def test_full_pipeline():
    """Test full pipeline from data to prediction"""
    logger.info("\nTesting full pipeline...")
    
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    
    # Get market data
    handler = MarketDataHandler(config.SYMBOL, config.TIMEFRAME)
    features = handler.get_market_features(config.BARS_TO_LOAD, config.FEATURES)
    
    if features is None:
        logger.error("Failed to get market features")
        mt5.shutdown()
        return False
    
    logger.info(f"Retrieved features shape: {features.shape}")
    
    # Initialize model
    num_features = len(config.FEATURES)
    model_manager = ModelManager(
        config.MODEL_NAME, 
        num_features, 
        config.SEQUENCE_LENGTH
    )
    
    # Get last sequence for prediction
    if len(features) >= config.SEQUENCE_LENGTH:
        input_features = features[-config.SEQUENCE_LENGTH:]
        signal, confidence = model_manager.predict_signal(input_features)
        
        signals = {0: "SELL", 1: "HOLD", 2: "BUY"}
        logger.info(f"\nCurrent market signal: {signals[signal]}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"Threshold: {config.PREDICTION_THRESHOLD}")
        
        if confidence >= config.PREDICTION_THRESHOLD:
            logger.info(f"✓ Signal strength sufficient for trading")
        else:
            logger.info(f"✗ Signal strength below trading threshold")
        
        logger.info("Full pipeline test PASSED")
    else:
        logger.error(f"Insufficient data: {len(features)} < {config.SEQUENCE_LENGTH}")
    
    mt5.shutdown()
    return True


def print_configuration():
    """Print current configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("CURRENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Symbol: {config.SYMBOL}")
    logger.info(f"Timeframe: {config.TIMEFRAME}")
    logger.info(f"Lot Size: {config.LOT_SIZE}")
    logger.info(f"Magic Number: {config.MAGIC_NUMBER}")
    logger.info(f"Stop Loss: {config.STOP_LOSS_PIPS} pips")
    logger.info(f"Take Profit: {config.TAKE_PROFIT_PIPS} pips")
    logger.info(f"Max Trades: {config.MAX_TRADES}")
    logger.info(f"Prediction Threshold: {config.PREDICTION_THRESHOLD}")
    logger.info(f"Sequence Length: {config.SEQUENCE_LENGTH}")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Features: {', '.join(config.FEATURES)}")
    logger.info("=" * 60 + "\n")


def main():
    """Run all tests"""
    print_configuration()
    
    logger.info("Starting XAU-EA-MT5 Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("Market Data", test_market_data),
        ("Transformer Model", test_transformer_model),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info('=' * 60)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED with exception: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
