from pathlib import Path
import json
import subprocess
import sys

def test_supervised_demo_runs(tmp_path: Path):
    outdir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "lab.lab_13_01_supervised_learning", "--demo", "--outdir", str(outdir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["unit"] == "13"
    assert "outputs" in metrics and len(metrics["outputs"]) >= 2

def test_unsupervised_demo_runs(tmp_path: Path):
    outdir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "lab.lab_13_02_unsupervised_learning", "--demo", "--outdir", str(outdir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    summary = json.loads((outdir / "unsupervised.json").read_text(encoding="utf-8"))
    assert summary["unit"] == "13"
