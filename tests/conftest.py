import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Tiny synthetic dataset + a 3-epoch model, shared by all tests."""
    base = tmp_path_factory.mktemp("pipeline")
    data, model, results = base / "data", base / "model", base / "results"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run([sys.executable, os.path.join(ROOT, "datasets/scripts/generate_synthetic.py"),
                    "--out", str(data), "--cells", "120", "--genes", "300", "--example-dir", str(base)],
                   check=True, env=env)
    subprocess.run([sys.executable, "-m", "ai_model.train", "--data-dir", str(data), "--model-dir", str(model),
                    "--output-dir", str(results), "--epochs", "3", "--top-k-expression", "200"],
                   check=True, env=env, cwd=ROOT)
    return {"data": data, "model": model, "results": results, "sample": base / "sample_patient.csv"}
