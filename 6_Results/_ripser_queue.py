# Shim: keep `python 6_Results/_ripser_queue.py` working after the move to Run_Queue/.
# The previous Ripser queue already finished; this file only forwards.
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "Run_Queue" / "_ripser_queue.py"),
    run_name="__main__",
)
