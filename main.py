"""
Main Expert Advisor using Transformer model for XAUUSD trading in MT5
"""

import MetaTrader5 as mt5
import time
import logging
import numpy as np
from datetime import datetime
import sys
import os

# Import EA components
from market_data import MarketDataHandler
from trade_manager import TradeManager
from transformer_model import ModelManager
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformerEA:
    """
    Expert Advisor using Transformer model for trading decisions
    """
    
    def __init__(self):
        """Initialize the Expert Advisor"""
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME
        self.lot_size = config.LOT_SIZE
        self.magic_number = config.MAGIC_NUMBER
        self.stop_loss = config.STOP_LOSS_PIPS
        self.take_profit = config.TAKE_PROFIT_PIPS
        self.max_trades = config.MAX_TRADES
        self.prediction_threshold = config.PREDICTION_THRESHOLD
        self.sequence_length = config.SEQUENCE_LENGTH
        
        # Initialize components
        self.market_data = None
        self.trade_manager = None
        self.model_manager = None
        
        self.is_initialized = False
        self.running = True
        
    def initialize(self):
        """Initialize MT5 connection and EA components"""
        logger.info("Initializing Transformer EA...")
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        logger.info(f"MT5 initialized: {mt5.version()}")
        
        # Login if credentials provided
        if (config.MT5_LOGIN is not None and 
            config.MT5_PASSWORD is not None and 
            config.MT5_SERVER is not None):
            if not mt5.login(config.MT5_LOGIN, config.MT5_PASSWORD, config.MT5_SERVER):
                logger.error("MT5 login failed")
                mt5.shutdown()
                return False
            logger.info("MT5 login successful")
        
        # Initialize market data handler
        self.market_data = MarketDataHandler(self.symbol, self.timeframe)
        
        # Initialize trade manager
        self.trade_manager = TradeManager(
            self.symbol, self.magic_number, self.lot_size
        )
        
        if not self.trade_manager.initialize():
            logger.error("Trade manager initialization failed")
            mt5.shutdown()
            return False
        
        # Initialize model manager
        num_features = len(config.FEATURES)
        self.model_manager = ModelManager(
            num_features, self.sequence_length
        )
        
        # Try to load existing model or initialize new one
        model_path = 'transformer_ea_model.pth'
        if os.path.exists(model_path):
            try:
                self.model_manager.load_model(model_path)
                logger.info("Loaded existing model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Will use untrained model.")
        else:
            logger.info("Using new model (untrained)")
        
        self.is_initialized = True
        logger.info("EA initialization completed successfully")
        return True
    
    def get_trading_signal(self):
        """
        Get trading signal from transformer model
        
        Returns:
            signal: 0=SELL, 1=HOLD, 2=BUY
            confidence: Prediction confidence
        """
        # Get market features
        features = self.market_data.get_market_features(
            config.BARS_TO_LOAD, config.FEATURES
        )
        
        if features is None or len(features) < self.sequence_length:
            logger.warning("Insufficient data for prediction")
            return 1, 0.0  # HOLD with 0 confidence
        
        # Use last sequence_length bars for prediction
        input_features = features[-self.sequence_length:]
        
        # Get prediction from model
        signal, confidence = self.model_manager.predict_signal(input_features)
        
        logger.info(f"Signal: {self._signal_to_str(signal)}, Confidence: {confidence:.4f}")
        
        return signal, confidence
    
    def _signal_to_str(self, signal):
        """Convert signal number to string"""
        signals = {0: "SELL", 1: "HOLD", 2: "BUY"}
        return signals.get(signal, "UNKNOWN")
    
    def execute_trading_logic(self):
        """Main trading logic"""
        # Check if max trades reached
        open_positions = self.trade_manager.count_open_positions()
        
        if open_positions >= self.max_trades:
            logger.debug(f"Max trades reached: {open_positions}/{self.max_trades}")
            return
        
        # Get trading signal
        signal, confidence = self.get_trading_signal()
        
        # Only trade if confidence is above threshold
        if confidence < self.prediction_threshold:
            logger.debug(f"Confidence {confidence:.4f} below threshold {self.prediction_threshold}")
            return
        
        # Execute trades based on signal
        if signal == 2:  # BUY
            logger.info(f"Opening BUY position (confidence: {confidence:.4f})")
            result = self.trade_manager.open_buy_position(
                self.stop_loss, self.take_profit
            )
            if result:
                logger.info(f"BUY order placed successfully")
                
        elif signal == 0:  # SELL
            logger.info(f"Opening SELL position (confidence: {confidence:.4f})")
            result = self.trade_manager.open_sell_position(
                self.stop_loss, self.take_profit
            )
            if result:
                logger.info(f"SELL order placed successfully")
                
        else:  # HOLD
            logger.debug("Signal is HOLD, no action taken")
    
    def print_status(self):
        """Print current EA status"""
        account_info = self.trade_manager.get_account_info()
        positions = self.trade_manager.get_open_positions()
        
        logger.info("=" * 50)
        logger.info(f"EA Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbol: {self.symbol}")
        
        if account_info:
            logger.info(f"Balance: ${account_info['balance']:.2f}")
            logger.info(f"Equity: ${account_info['equity']:.2f}")
            logger.info(f"Profit: ${account_info['profit']:.2f}")
        
        logger.info(f"Open Positions: {len(positions)}")
        
        for i, pos in enumerate(positions, 1):
            pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            logger.info(f"  Position {i}: {pos_type} {pos.volume} lots, "
                       f"Profit: ${pos.profit:.2f}")
        
        logger.info("=" * 50)
    
    def run(self):
        """Main EA loop"""
        if not self.is_initialized:
            logger.error("EA not initialized. Call initialize() first.")
            return
        
        logger.info("Starting EA main loop...")
        logger.info(f"Trading: {self.symbol}, Timeframe: {self.timeframe}")
        logger.info(f"Lot Size: {self.lot_size}, Max Trades: {self.max_trades}")
        logger.info(f"Prediction Threshold: {self.prediction_threshold}")
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                
                # Print status periodically
                if iteration % 10 == 0:
                    self.print_status()
                
                # Execute trading logic
                try:
                    self.execute_trading_logic()
                except Exception as e:
                    logger.error(f"Error in trading logic: {e}", exc_info=True)
                
                # Wait before next iteration
                logger.debug(f"Waiting {config.PREDICTION_INTERVAL} seconds...")
                time.sleep(config.PREDICTION_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("EA stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown EA and cleanup"""
        logger.info("Shutting down EA...")
        
        self.running = False
        
        # Print final status
        self.print_status()
        
        # Save model if trained
        if self.model_manager and self.model_manager.is_trained:
            try:
                self.model_manager.save_model('transformer_ea_model.pth')
            except Exception as e:
                logger.error(f"Error saving model: {e}")
        
        # Shutdown MT5
        mt5.shutdown()
        logger.info("EA shutdown complete")


def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("XAU-EA-MT5: Transformer-based Expert Advisor")
    logger.info("=" * 50)
    
    # Create and initialize EA
    ea = TransformerEA()
    
    if not ea.initialize():
        logger.error("Failed to initialize EA. Exiting.")
        return 1
    
    # Run EA
    ea.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
