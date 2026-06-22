"""
Shared pytest fixtures for DRI signal construction tests.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
