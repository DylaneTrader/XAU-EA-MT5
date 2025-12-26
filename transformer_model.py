"""
Transformer-based Price Prediction Model for MT5 EA
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoConfig
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerPricePredictor(nn.Module):
    """
    Transformer model for price prediction using pretrained language model
    adapted for time series forecasting
    """
    
    def __init__(self, model_name, input_dim, hidden_dim=128, num_labels=3):
        """
        Initialize the Transformer predictor
        
        Args:
            model_name: Name of pretrained transformer model
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_labels: Number of output classes (BUY, SELL, HOLD)
        """
        super(TransformerPricePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer to match transformer dimensions
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim // 2, num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        logger.info(f"Initialized Transformer model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output predictions of shape (batch_size, num_labels)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch, seq, hidden_dim)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq, hidden_dim)
        
        # Use the last time step's output
        x = x[:, -1, :]  # (batch, hidden_dim)
        
        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.softmax(x)
    
    def predict(self, x):
        """
        Make predictions (BUY=2, HOLD=1, SELL=0)
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class and confidence
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            output = self.forward(x)
            confidence, prediction = torch.max(output, dim=1)
            
            return prediction.item(), confidence.item()


class ModelManager:
    """Manager for training and using the Transformer model"""
    
    def __init__(self, model_name, input_dim, sequence_length):
        """
        Initialize the model manager
        
        Args:
            model_name: Transformer model name
            input_dim: Number of input features
            sequence_length: Length of input sequences
        """
        self.model = TransformerPricePredictor(model_name, input_dim)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.is_trained = False
        
    def prepare_features(self, data):
        """
        Prepare and normalize features
        
        Args:
            data: Input data array
            
        Returns:
            Normalized features
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Only transform data, never fit on prediction data
        # Scaler should be fitted during training only
        return self.scaler.transform(data)
    
    def train_simple(self, X, y, epochs=10, lr=0.001):
        """
        Simple training loop for the model
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            lr: Learning rate
        """
        # Fit scaler on training data
        if not self.is_trained:
            # Reshape X for scaler fitting: (samples, seq_len, features) -> (samples*seq_len, features)
            X_reshaped = X.reshape(-1, X.shape[-1])
            self.scaler.fit(X_reshaped)
            logger.info("Scaler fitted on training data")
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        self.is_trained = True
        logger.info("Model training completed")
    
    def predict_signal(self, features):
        """
        Predict trading signal
        
        Args:
            features: Input features array
            
        Returns:
            Signal (0=SELL, 1=HOLD, 2=BUY) and confidence
        """
        features_normalized = self.prepare_features(features)
        prediction, confidence = self.model.predict(features_normalized)
        
        return prediction, confidence
    
    def save_model(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.is_trained = checkpoint['is_trained']
        logger.info(f"Model loaded from {path}")
