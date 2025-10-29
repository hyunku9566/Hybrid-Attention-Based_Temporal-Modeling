#!/usr/bin/env python3
"""
Wrapper script to run training from project root

This allows proper module imports when running from the project root directory.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the train module
if __name__ == '__main__':
    from train import train
    train.main()
