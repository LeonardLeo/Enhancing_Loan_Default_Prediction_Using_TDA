# -*- coding: utf-8 -*-
"""Convenience launcher. Open `pkdd_czech_financial_audit.py` in this folder to read the method."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name('pkdd_czech_financial_audit.py')), run_name="__main__")
