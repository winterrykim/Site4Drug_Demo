import subprocess
import sys
from pathlib import Path


def test_predict_site_module_help_runs_cleanly():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "site4drug_inference.demo.predict_site", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "Run Site4Drug modality-aware prediction" in result.stdout
    assert "RuntimeWarning" not in result.stderr
