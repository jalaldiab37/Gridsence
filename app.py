"""
GridSense Dashboard - Streamlit Cloud Entry Point
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()
else:
    # When imported by Streamlit
    main()

