"""
Streamlit Dashboard for XAU-EA-MT5 Model Training and Evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import io
import os
from datetime import datetime

# Import your existing modules
from transformer_model import TransformerPricePredictor
import config

# Set page configuration
st.set_page_config(
    page_title="XAU-EA-MT5 Training Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def calculate_indicators(df):
    """Calculate technical indicators using TA library"""
    import ta
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close']
    ).average_true_range()
    
    # Drop NaN values created by indicators
    df = df.dropna()
    
    return df


def create_labels(df, forward_bars=5, threshold=0.001):
    """Create labels based on future price movement"""
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
    
    # Pad with HOLD for the last forward_bars
    labels.extend([1] * forward_bars)
    
    return np.array(labels)


def prepare_features(df, feature_columns):
    """Prepare and normalize features"""
    features = df[feature_columns].values
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized, scaler


def prepare_sequences(features, labels, sequence_length):
    """Prepare sequences for training"""
    X = []
    y = []
    
    for i in range(sequence_length, len(features)):
        if i < len(labels):
            X.append(features[i-sequence_length:i])
            y.append(labels[i])
    
    return np.array(X), np.array(y)


def train_model_fn(model, X_train, y_train, epochs, device):
    """Train the model with progress tracking"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    # Training loop
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor)
        
        # Store metrics
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
    
    return model, history


def evaluate_model(model, X_test, y_test, device):
    """Evaluate model on test set"""
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs.data, 1)
        confidences = torch.max(outputs.data, 1)[0]
    
    predictions = predictions.cpu().numpy()
    confidences = confidences.cpu().numpy()
    
    return predictions, confidences


# Main Dashboard
def main():
    st.markdown('<div class="main-header">📈 XAU-EA-MT5 Training Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    st.sidebar.subheader("1. Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload historical data (CSV/XLSX)", 
        type=['csv', 'xlsx']
    )
    
    # Training parameters
    st.sidebar.subheader("2. Training Parameters")
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 20, 5)
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60, 10)
    forward_bars = st.sidebar.slider("Forward Bars (Label)", 3, 10, 5, 1)
    threshold = st.sidebar.slider("Price Threshold", 0.0001, 0.005, 0.001, 0.0001, format="%.4f")
    
    # Model parameters
    st.sidebar.subheader("3. Model Parameters")
    hidden_dim = st.sidebar.selectbox("Hidden Dimension", [64, 128, 256], index=1)
    num_layers = st.sidebar.selectbox("Transformer Layers", [2, 4, 6, 8], index=1)
    
    # Save path
    st.sidebar.subheader("4. Model Save")
    model_save_path = st.sidebar.text_input("Model Save Path", "transformer_ea_model.pth")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Training", "🎯 Evaluation", "💾 Model Management"])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<div class="section-header">Data Loading & Preprocessing</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Load data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} bars of data")
                
                # Validate required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("Required columns: open, high, low, close, volume")
                    return
                
                # Display data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Bars", len(df))
                with col2:
                    st.metric("Features", len(df.columns))
                with col3:
                    st.metric("Date Range", f"{len(df)} bars")
                
                # Show sample data
                st.subheader("Sample Data (First 10 rows)")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check for minimum bars
                if len(df) < 5000:
                    st.warning(f"⚠️ Data has {len(df)} bars. Recommended: 5000+ bars for better training.")
                
                # Calculate indicators
                if st.button("🔧 Calculate Technical Indicators", type="primary"):
                    with st.spinner("Calculating indicators..."):
                        df_with_indicators = calculate_indicators(df.copy())
                        st.session_state.df = df_with_indicators
                        st.session_state.data_loaded = True
                        st.success(f"✅ Indicators calculated! Data shape: {df_with_indicators.shape}")
                        st.dataframe(df_with_indicators.head(10), use_container_width=True)
                
                # If indicators are calculated
                if st.session_state.data_loaded:
                    st.subheader("📊 Technical Indicators")
                    df_display = st.session_state.df
                    
                    # Plot some indicators
                    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
                    
                    # Price and Bollinger Bands
                    axes[0].plot(df_display['close'][-200:], label='Close Price')
                    axes[0].plot(df_display['bb_upper'][-200:], label='BB Upper', linestyle='--')
                    axes[0].plot(df_display['bb_lower'][-200:], label='BB Lower', linestyle='--')
                    axes[0].set_title('Price with Bollinger Bands (Last 200 bars)')
                    axes[0].legend()
                    axes[0].grid(True)
                    
                    # RSI
                    axes[1].plot(df_display['rsi'][-200:], label='RSI', color='orange')
                    axes[1].axhline(y=70, color='r', linestyle='--', label='Overbought')
                    axes[1].axhline(y=30, color='g', linestyle='--', label='Oversold')
                    axes[1].set_title('RSI (Last 200 bars)')
                    axes[1].legend()
                    axes[1].grid(True)
                    
                    # MACD
                    axes[2].plot(df_display['macd'][-200:], label='MACD')
                    axes[2].plot(df_display['macd_signal'][-200:], label='Signal')
                    axes[2].set_title('MACD (Last 200 bars)')
                    axes[2].legend()
                    axes[2].grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        else:
            st.info("👆 Upload a CSV or XLSX file to get started")
            st.markdown("""
            **Required columns:**
            - `open`: Opening price
            - `high`: High price
            - `low`: Low price
            - `close`: Closing price
            - `volume`: Trading volume
            
            **Recommended:** 5000+ bars for optimal training
            """)
    
    # Tab 2: Training
    with tab2:
        st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load and process data first in the 'Data Overview' tab")
        else:
            df = st.session_state.df
            
            # Feature selection
            st.subheader("Feature Selection")
            available_features = ['open', 'high', 'low', 'close', 'volume', 
                                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr']
            
            # Filter available features based on what's in the dataframe
            available_features = [f for f in available_features if f in df.columns]
            
            selected_features = st.multiselect(
                "Select features for training",
                available_features,
                default=available_features
            )
            
            if len(selected_features) < 3:
                st.warning("⚠️ Please select at least 3 features")
                return
            
            # Train button
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Preparing data..."):
                    # Create labels
                    st.info(f"Creating labels with forward_bars={forward_bars}, threshold={threshold}")
                    labels = create_labels(df, forward_bars=forward_bars, threshold=threshold)
                    
                    # Display label distribution
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BUY Labels", np.sum(labels == 2))
                    with col2:
                        st.metric("HOLD Labels", np.sum(labels == 1))
                    with col3:
                        st.metric("SELL Labels", np.sum(labels == 0))
                    
                    # Prepare features
                    st.info("Preparing and normalizing features...")
                    features, scaler = prepare_features(df, selected_features)
                    
                    # Create sequences
                    st.info(f"Creating sequences with length={sequence_length}")
                    X, y = prepare_sequences(features, labels, sequence_length)
                    
                    st.success(f"✅ Prepared {len(X)} sequences")
                    
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, shuffle=False
                    )
                    
                    st.info(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.selected_features = selected_features
                
                # Initialize model
                with st.spinner("Initializing model..."):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    st.info(f"Using device: {device}")
                    
                    num_features = len(selected_features)
                    model = TransformerPricePredictor(
                        input_dim=num_features,
                        hidden_dim=hidden_dim,
                        num_labels=3
                    ).to(device)
                    
                    st.session_state.model = model
                    st.session_state.device = device
                
                # Train model
                st.subheader("Training Progress")
                model, history = train_model_fn(
                    model, X_train, y_train, epochs, device
                )
                
                st.session_state.model = model
                st.session_state.history = history
                st.session_state.model_trained = True
                
                st.success("✅ Training completed!")
                
                # Plot training history
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                axes[0].plot(history['loss'])
                axes[0].set_title('Training Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].grid(True)
                
                axes[1].plot(history['accuracy'])
                axes[1].set_title('Training Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # Tab 3: Evaluation
    with tab3:
        st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Training' tab")
        else:
            model = st.session_state.model
            device = st.session_state.device
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Evaluate model
            with st.spinner("Evaluating model on test set..."):
                predictions, confidences = evaluate_model(model, X_test, y_test, device)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                
                # Display metrics
                st.subheader("📊 Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Test Samples", len(y_test))
                with col3:
                    avg_confidence = np.mean(confidences)
                    st.metric("Avg Confidence", f"{avg_confidence:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['SELL', 'HOLD', 'BUY'],
                           yticklabels=['SELL', 'HOLD', 'BUY'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                plt.close()
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, predictions, 
                                              target_names=['SELL', 'HOLD', 'BUY'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Prediction distribution
                st.subheader("Prediction Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    labels_count = [np.sum(y_test == 0), np.sum(y_test == 1), np.sum(y_test == 2)]
                    ax.bar(['SELL', 'HOLD', 'BUY'], labels_count, color=['red', 'gray', 'green'])
                    ax.set_title('True Labels Distribution')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pred_count = [np.sum(predictions == 0), np.sum(predictions == 1), np.sum(predictions == 2)]
                    ax.bar(['SELL', 'HOLD', 'BUY'], pred_count, color=['red', 'gray', 'green'])
                    ax.set_title('Predicted Labels Distribution')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    plt.close()
                
                # Store results
                st.session_state.predictions = predictions
                st.session_state.confidences = confidences
    
    # Tab 4: Model Management
    with tab4:
        st.markdown('<div class="section-header">Model Management</div>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Training' tab")
        else:
            model = st.session_state.model
            
            # Save model
            st.subheader("💾 Save Model")
            
            if st.button("Save Model to Disk", type="primary"):
                try:
                    # Prepare model info
                    model_info = {
                        'model_state_dict': model.state_dict(),
                        'features': st.session_state.selected_features,
                        'scaler_mean': st.session_state.scaler.mean_.tolist(),
                        'scaler_scale': st.session_state.scaler.scale_.tolist(),
                        'sequence_length': sequence_length,
                        'hidden_dim': hidden_dim,
                        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'test_accuracy': accuracy_score(st.session_state.y_test, st.session_state.predictions)
                    }
                    
                    # Save model
                    torch.save(model_info, model_save_path)
                    st.success(f"✅ Model saved successfully to: {model_save_path}")
                    
                    # Display model info
                    st.info(f"""
                    **Model Information:**
                    - Features: {len(st.session_state.selected_features)}
                    - Sequence Length: {sequence_length}
                    - Hidden Dimension: {hidden_dim}
                    - Test Accuracy: {model_info['test_accuracy']:.4f}
                    - Saved: {model_info['training_date']}
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
            
            # Model info
            st.subheader("📋 Current Model Information")
            if 'selected_features' in st.session_state:
                st.write("**Selected Features:**")
                st.write(st.session_state.selected_features)
            
            st.write("**Model Architecture:**")
            st.write(f"- Input Dimension: {len(st.session_state.selected_features) if 'selected_features' in st.session_state else 'N/A'}")
            st.write(f"- Hidden Dimension: {hidden_dim}")
            st.write(f"- Sequence Length: {sequence_length}")
            st.write(f"- Output Classes: 3 (SELL, HOLD, BUY)")
            
            # Download model
            st.subheader("⬇️ Download Model")
            if os.path.exists(model_save_path):
                with open(model_save_path, 'rb') as f:
                    model_bytes = f.read()
                
                st.download_button(
                    label="Download Model File",
                    data=model_bytes,
                    file_name=model_save_path,
                    mime="application/octet-stream"
                )


if __name__ == "__main__":
    main()
