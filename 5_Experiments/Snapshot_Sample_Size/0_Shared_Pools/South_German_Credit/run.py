# -*- coding: utf-8 -*-
"""Convenience launcher. Open `south_german_credit_shared_pools.py` in this folder to read the method."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name('south_german_credit_shared_pools.py')), run_name="__main__")
