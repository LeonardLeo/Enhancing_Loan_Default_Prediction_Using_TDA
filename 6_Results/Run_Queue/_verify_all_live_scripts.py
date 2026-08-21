# One-shot: execute every live dataset script and record pass/fail.
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = HERE / "_verify_all_live_scripts.log"
SUMMARY = HERE / "_verify_all_live_scripts_summary.txt"

SKIP_NAMES = {
    "run.py",
    "run_all.py",
    "run_protocol.py",
    "protocol_lib.py",
    "sample_size_lib.py",
    "_protocol_launcher_template.py",
}
SKIP_DIR_PARTS = {"Archives", "Default_Parameters", "__pycache__"}

FAST_TIMEOUT = 240
TUNED_TIMEOUT = 1200
PROTOCOL_TIMEOUT = 1800
SAMPLE_TIMEOUT = 600
VIS_TIMEOUT = 300


def collect_scripts():
    scripts = []
    experiments = ROOT / "5_Experiments"
    for dirpath, dirnames, files in os.walk(experiments):
        parts = set(Path(dirpath).parts)
        if SKIP_DIR_PARTS & parts:
            continue
        for name in files:
            if not name.endswith(".py") or name in SKIP_NAMES:
                continue
            scripts.append(Path(dirpath) / name)
    visualize = list((ROOT / "5_Experiments" / "Snapshot_Sample_Size").glob("*/visualize_results.py"))
    return scripts, visualize


def timeout_for(path: Path) -> int:
    name = path.name
    if "visualize_results" in name:
        return VIS_TIMEOUT
    if name.endswith("_PH_tuned.py") or name.endswith("_PH_CV.py"):
        return TUNED_TIMEOUT
    if name.endswith("_protocol.py"):
        return PROTOCOL_TIMEOUT
    if name.endswith("_sample_size.py"):
        return SAMPLE_TIMEOUT
    return FAST_TIMEOUT


def sort_key(path: Path):
    name = path.name
    if name.endswith("_protocol.py"):
        group = 4
    elif name.endswith("_PH_tuned.py") or name.endswith("_PH_CV.py"):
        group = 3
    elif name.endswith("_sample_size.py"):
        group = 2
    elif "visualize_results" in name:
        group = 5
    else:
        group = 1
    return (group, str(path).lower())


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def already_passed(rel: str) -> bool:
    if not LOG.exists():
        return False
    needle = f" PASS exit=0 "
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if needle in line and line.endswith(rel):
            return True
    return False


def win_script(path: Path) -> str:
    raw = os.path.abspath(path)
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        raw = "\\\\?\\" + raw
    return raw


def run_one(path: Path):
    rel = path.relative_to(ROOT).as_posix()
    limit = timeout_for(path)
    started = time.time()
    try:
        completed = subprocess.run(
            [str(PY), win_script(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=limit,
        )
        elapsed = time.time() - started
        ok = completed.returncode == 0
        tail = (completed.stdout or "")[-1500:]
        err = (completed.stderr or "")[-2000:]
        return rel, ok, completed.returncode, elapsed, tail, err
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        err = (exc.stderr or b"").decode("utf-8", errors="replace")[-2000:] if exc.stderr else "TIMEOUT"
        return rel, False, -9, elapsed, "", err


def main():
    scripts, visualize = collect_scripts()
    ordered = sorted(scripts + visualize, key=sort_key)
    log(f"Resume/run {len(ordered)} live scripts with {PY}")
    passed = []
    failed = []
    skipped = []
    for index, path in enumerate(ordered, start=1):
        rel = path.relative_to(ROOT).as_posix()
        if already_passed(rel):
            log(f"({index}/{len(ordered)}) SKIP already passed {rel}")
            skipped.append(rel)
            passed.append((rel, 0.0))
            continue
        log(f"({index}/{len(ordered)}) START {rel} timeout={timeout_for(path)}s")
        rel, ok, code, elapsed, tail, err = run_one(path)
        status = "PASS" if ok else "FAIL"
        log(f"({index}/{len(ordered)}) {status} exit={code} {elapsed:.1f}s {rel}")
        if not ok:
            if tail:
                log("STDOUT_TAIL " + tail.replace("\n", " | "))
            if err:
                log("STDERR_TAIL " + err.replace("\n", " | "))
            failed.append((rel, code, elapsed))
        else:
            passed.append((rel, elapsed))
    lines = [
        f"passed {len(passed)} (skipped already-passed {len(skipped)})",
        f"failed {len(failed)}",
        f"total {len(ordered)}",
        "",
        "FAILED:" if failed else "FAILED: none",
    ]
    for rel, code, elapsed in failed:
        lines.append(f"  exit={code} {elapsed:.1f}s {rel}")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("DONE " + " | ".join(lines[:3]))
    print(SUMMARY.read_text(encoding="utf-8"), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
