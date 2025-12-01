#!/usr/bin/env python
"""
Quick start script for GridSense demo.
Run this file to see the system in action without any setup.
"""

import sys
from pathlib import Path

# Add gridsense to path
sys.path.insert(0, str(Path(__file__).parent))

from main import run_demo

if __name__ == "__main__":
    run_demo()


