"""
Training script for the Transformer model using historical data
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

from market_data import MarketDataHandler
from transformer_model import ModelManager
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_labels(df, forward_bars=5, threshold=0.001):
    """
    Create labels based on future price movement
    
    Args:
        df: DataFrame with OHLCV data
        forward_bars: Number of bars to look ahead
        threshold: Price change threshold for BUY/SELL signals
        
    Returns:
        Array of labels (0=SELL, 1=HOLD, 2=BUY)
    """
    labels = []
    
    for i in range(len(df) - forward_bars):
        current_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + forward_bars]
        
        price_change = (future_price - current_price) / current_price
        
        if price_change > threshold:
            labels.append(2)  # BUY
        elif price_change < -threshold:
            labels.append(0)  # SELL
        else:
            labels.append(1)  # HOLD
    
    return np.array(labels)


def prepare_training_data(features, labels, sequence_length):
    """
    Prepare sequences for training
    
    Args:
        features: Feature matrix
        labels: Label array
        sequence_length: Length of input sequences
        
    Returns:
        X (sequences), y (labels)
    """
    X = []
    y = []
    
    for i in range(sequence_length, len(features)):
        if i < len(labels):
            X.append(features[i-sequence_length:i])
            y.append(labels[i])
    
    return np.array(X), np.array(y)


def train_model(num_bars=5000, epochs=20, test_size=0.2):
    """
    Train the Transformer model on historical data
    
    Args:
        num_bars: Number of historical bars to use
        epochs: Number of training epochs
        test_size: Proportion of data for testing
    """
    logger.info("Starting model training...")
    logger.info("=" * 60)
    
    # Initialize MT5
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    
    logger.info(f"MT5 initialized: {mt5.version()}")
    
    # Get market data
    logger.info(f"Fetching {num_bars} bars of historical data...")
    handler = MarketDataHandler(config.SYMBOL, config.TIMEFRAME)
    df = handler.get_historical_data(num_bars)
    
    if df is None:
        logger.error("Failed to retrieve market data")
        mt5.shutdown()
        return False
    
    logger.info(f"Retrieved {len(df)} bars")
    
    # Calculate indicators
    logger.info("Calculating technical indicators...")
    df = handler.calculate_indicators(df)
    
    if df is None or len(df) < config.SEQUENCE_LENGTH + 10:
        logger.error("Insufficient data after indicator calculation")
        mt5.shutdown()
        return False
    
    logger.info(f"Indicators calculated for {len(df)} bars")
    
    # Prepare features
    features = handler.prepare_features(df, config.FEATURES)
    
    if features is None:
        logger.error("Failed to prepare features")
        mt5.shutdown()
        return False
    
    logger.info(f"Features shape: {features.shape}")
    
    # Create labels
    logger.info("Creating labels based on future price movement...")
    labels = create_labels(df, forward_bars=5, threshold=0.0005)
    
    logger.info(f"Labels created: {len(labels)}")
    logger.info(f"Label distribution - BUY: {np.sum(labels==2)}, "
                f"HOLD: {np.sum(labels==1)}, SELL: {np.sum(labels==0)}")
    
    # Prepare sequences
    logger.info("Preparing training sequences...")
    X, y = prepare_training_data(features, labels, config.SEQUENCE_LENGTH)
    
    logger.info(f"Training data shape - X: {X.shape}, y: {y.shape}")
    
    # Split data (no shuffle for time series, no random_state needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize model
    logger.info("\nInitializing Transformer model...")
    num_features = len(config.FEATURES)
    model_manager = ModelManager(
        config.MODEL_NAME, 
        num_features, 
        config.SEQUENCE_LENGTH
    )
    
    # Train model
    logger.info(f"\nTraining model for {epochs} epochs...")
    logger.info("=" * 60)
    model_manager.train_simple(X_train, y_train, epochs=epochs)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    correct = 0
    total = len(X_test)
    
    predictions = []
    for i in range(total):
        pred, conf = model_manager.predict_signal(X_test[i])
        predictions.append(pred)
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Prediction distribution
    predictions = np.array(predictions)
    logger.info(f"Prediction distribution - BUY: {np.sum(predictions==2)}, "
                f"HOLD: {np.sum(predictions==1)}, SELL: {np.sum(predictions==0)}")
    
    # Save model
    model_path = 'transformer_ea_model.pth'
    logger.info(f"\nSaving model to {model_path}...")
    model_manager.save_model(model_path)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)
    
    mt5.shutdown()
    return True


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("XAU-EA-MT5: Model Training Script")
    logger.info("=" * 60)
    logger.info(f"Symbol: {config.SYMBOL}")
    logger.info(f"Timeframe: {config.TIMEFRAME}")
    logger.info(f"Sequence Length: {config.SEQUENCE_LENGTH}")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info("=" * 60)
    
    success = train_model(num_bars=5000, epochs=20)
    
    if success:
        logger.info("\n✓ Model training successful!")
        logger.info("You can now run main.py to start the EA with the trained model.")
        return 0
    else:
        logger.error("\n✗ Model training failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
