"""
Trading manager for executing and managing trades
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeManager:
    """Manager for executing and monitoring trades"""
    
    def __init__(self, symbol, magic_number, lot_size):
        """
        Initialize trade manager
        
        Args:
            symbol: Trading symbol
            magic_number: Unique identifier for EA trades
            lot_size: Trading lot size
        """
        self.symbol = symbol
        self.magic_number = magic_number
        self.lot_size = lot_size
        self.symbol_info = None
        
    def initialize(self):
        """Initialize and get symbol information"""
        self.symbol_info = mt5.symbol_info(self.symbol)
        
        if self.symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False
        
        if not self.symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select {self.symbol}")
                return False
        
        logger.info(f"Symbol {self.symbol} initialized")
        return True
    
    def get_open_positions(self):
        """Get all open positions for this EA"""
        positions = mt5.positions_get(symbol=self.symbol)
        
        if positions is None:
            return []
        
        # Filter by magic number
        ea_positions = [p for p in positions if p.magic == self.magic_number]
        return ea_positions
    
    def count_open_positions(self):
        """Count open positions"""
        return len(self.get_open_positions())
    
    def open_buy_position(self, stop_loss_pips, take_profit_pips, comment="Transformer EA Buy"):
        """
        Open a BUY position
        
        Args:
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            comment: Order comment
            
        Returns:
            Order result
        """
        if self.symbol_info is None:
            logger.error("Symbol info not initialized")
            return None
        
        price = mt5.symbol_info_tick(self.symbol).ask
        point = self.symbol_info.point
        
        # Calculate SL and TP
        sl = price - stop_loss_pips * point * 10
        tp = price + take_profit_pips * point * 10
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Buy order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Buy order successful: {result.order} at {price}")
        return result
    
    def open_sell_position(self, stop_loss_pips, take_profit_pips, comment="Transformer EA Sell"):
        """
        Open a SELL position
        
        Args:
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            comment: Order comment
            
        Returns:
            Order result
        """
        if self.symbol_info is None:
            logger.error("Symbol info not initialized")
            return None
        
        price = mt5.symbol_info_tick(self.symbol).bid
        point = self.symbol_info.point
        
        # Calculate SL and TP
        sl = price + stop_loss_pips * point * 10
        tp = price - take_profit_pips * point * 10
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Sell order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Sell order successful: {result.order} at {price}")
        return result
    
    def close_position(self, position):
        """
        Close a specific position
        
        Args:
            position: Position object to close
            
        Returns:
            Order result
        """
        tick = mt5.symbol_info_tick(self.symbol)
        
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "Close by EA",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Position closed: {position.ticket}")
        return result
    
    def close_all_positions(self):
        """Close all open positions for this EA"""
        positions = self.get_open_positions()
        
        for position in positions:
            self.close_position(position)
        
        logger.info(f"Closed {len(positions)} positions")
    
    def get_account_info(self):
        """Get account information"""
        account_info = mt5.account_info()
        
        if account_info is None:
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'profit': account_info.profit,
        }
