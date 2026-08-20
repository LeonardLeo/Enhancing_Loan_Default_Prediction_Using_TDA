# -*- coding: utf-8 -*-
"""Convenience launcher. Open `taiwan_bankruptcy_sample_size.py` in this folder to read the method."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name('taiwan_bankruptcy_sample_size.py')), run_name="__main__")
