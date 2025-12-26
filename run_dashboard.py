"""
Quick Start Script - Run this to launch the Streamlit dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'torch',
        'sklearn',
        'ta',
        'matplotlib',
        'seaborn',
        'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def main():
    print("=" * 60)
    print("🚀 XAU-EA-MT5 Streamlit Dashboard Launcher")
    print("=" * 60)
    print()
    
    # Check if requirements are installed
    print("Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All requirements satisfied!")
    print()
    
    # Check if streamlit_dashboard.py exists
    dashboard_file = "streamlit_dashboard.py"
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: {dashboard_file} not found!")
        sys.exit(1)
    
    # Launch Streamlit
    print("🌐 Launching Streamlit dashboard...")
    print("📝 The dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print()
    print("💡 Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_file])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
