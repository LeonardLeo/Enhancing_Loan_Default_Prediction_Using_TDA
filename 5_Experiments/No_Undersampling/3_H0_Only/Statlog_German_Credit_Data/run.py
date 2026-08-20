# -*- coding: utf-8 -*-
"""Convenience launcher. Open `statlog_german_credit_data_H0_only.py` in this folder to read the method."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name('statlog_german_credit_data_H0_only.py')), run_name="__main__")
