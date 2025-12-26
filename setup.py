#!/usr/bin/env python3
"""
Quick setup and installation script for XAU-EA-MT5
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is adequate"""
    logger.info("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages"""
    logger.info("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_mt5_installation():
    """Check if MT5 is accessible"""
    logger.info("\nChecking MetaTrader 5 connectivity...")
    
    try:
        import MetaTrader5 as mt5
        
        if mt5.initialize():
            version = mt5.version()
            logger.info(f"✓ MT5 connected: {version}")
            mt5.shutdown()
            return True
        else:
            logger.warning("⚠ MT5 initialized but connection failed")
            logger.warning("  This is normal if MT5 terminal is not running")
            return True
    except ImportError:
        logger.error("✗ MetaTrader5 package not found")
        return False
    except Exception as e:
        logger.warning(f"⚠ MT5 check failed: {e}")
        logger.warning("  This is normal if MT5 terminal is not running")
        return True


def verify_installation():
    """Verify all components are working"""
    logger.info("\nVerifying installation...")
    
    try:
        # Test imports
        logger.info("Testing imports...")
        import numpy
        import pandas
        import torch
        import transformers
        import sklearn
        import ta
        
        logger.info("✓ All required packages imported successfully")
        
        # Test model creation
        logger.info("Testing Transformer model...")
        from transformer_model import ModelManager
        model = ModelManager(11, 60)
        logger.info("✓ Model created successfully")
        
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def create_env_example():
    """Create .env.example file"""
    logger.info("\nCreating .env.example file...")
    
    env_content = """# MetaTrader 5 Configuration (Optional for demo accounts)
# Uncomment and fill in your credentials for live accounts

# MT5_LOGIN=your_login_here
# MT5_PASSWORD=your_password_here
# MT5_SERVER=your_server_here

# Example:
# MT5_LOGIN=12345678
# MT5_PASSWORD=MyPassword123
# MT5_SERVER=MetaQuotes-Demo
"""
    
    try:
        with open('.env.example', 'w') as f:
            f.write(env_content)
        logger.info("✓ Created .env.example")
        return True
    except Exception as e:
        logger.error(f"Failed to create .env.example: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Ensure MetaTrader 5 terminal is installed and running")
    logger.info("2. Enable algorithmic trading in MT5:")
    logger.info("   Tools > Options > Expert Advisors > Allow algorithmic trading")
    logger.info("3. Configure your settings in config.py")
    logger.info("4. (Optional) Create .env file with your MT5 credentials")
    logger.info("5. Run tests: python test_ea.py")
    logger.info("6. Train model (optional): python train_model.py")
    logger.info("7. Start EA: python main.py")
    logger.info("\n⚠ IMPORTANT: Always test on a DEMO account first!")
    logger.info("=" * 60)


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("XAU-EA-MT5 Setup Script")
    logger.info("=" * 60)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_requirements),
        ("Check MT5", check_mt5_installation),
        ("Verify Installation", verify_installation),
        ("Create Environment Example", create_env_example),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step: {step_name}")
        logger.info('=' * 60)
        
        if not step_func():
            logger.error(f"\n✗ Setup failed at: {step_name}")
            logger.error("Please fix the errors above and run setup again.")
            return 1
    
    print_next_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main())
