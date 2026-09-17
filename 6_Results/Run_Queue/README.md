# Run queue

Operational scripts and bookkeeping that do not belong inside one protocol bucket.

| File | What it is |
|------|------------|
| `_ripser_queue.py` | Sequential Ripser / protocol queue (resume-safe). |
| `_snapshot_sample_size_queue.py` | Resume-safe Snapshot_Sample_Size Ripser/ML queue (does not interleave the historical queue). |
| `_ripser_queue.log` | Log from the last Ripser queue run. |
| `_consumer_queue.py` | Resume-safe consumer trainer (Exp 1 models + Exp 2–8 when barcodes exist). |
| `_consumer_queue.log` | Log from the last consumer queue run. |
| `registry_*.json` / `registry_*.csv` | Run coverage, manifests, and source verification. |
| `_compile_results_briefing.py` | Builds the supervisor HTML briefing `6_Results/Compiled_Reports/TDA_Results_Briefing.html` from live Default of Credit Card Client artefacts (Plotly charts, worked substitutions, appendices A–D). The HTML is generated locally (root `.gitignore` ignores `*.html`). |
| `_compile_two_dataset_report.py` | Builds the table-only PDF `6_Results/Compiled_Reports/Default_Of_Credit_Card_Client_Results.pdf`. |

```powershell
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_ripser_queue.py
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_consumer_queue.py
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_snapshot_sample_size_queue.py
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_compile_results_briefing.py
```

Compatibility shims at `6_Results/_ripser_queue.py` and `6_Results/_consumer_queue.py` forward to these scripts so an old command still works.
