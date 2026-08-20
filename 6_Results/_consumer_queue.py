# Shim: keep `python 6_Results/_consumer_queue.py` working after the move to Run_Queue/.
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "Run_Queue" / "_consumer_queue.py"),
    run_name="__main__",
)
