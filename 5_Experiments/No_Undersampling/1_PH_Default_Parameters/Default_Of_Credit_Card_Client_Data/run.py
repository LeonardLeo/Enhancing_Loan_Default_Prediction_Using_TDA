# -*- coding: utf-8 -*-
"""Convenience launcher. Open `default_of_credit_cards_client_PH.py` in this folder to read the method."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name('default_of_credit_cards_client_PH.py')), run_name="__main__")
