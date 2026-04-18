import importlib.util
import json
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "simulation.py"
    spec = importlib.util.spec_from_file_location("aleph_point_synthesis_simulation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_outputs_to_requested_project_root(tmp_path):
    module = load_module()

    result = module.main(project_root=tmp_path, seed=726)

    certification_path = tmp_path / "certification.json"
    data_path = tmp_path / "data.csv"

    assert certification_path.exists()
    assert data_path.exists()

    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert certification["method"] == "Aleph-Point Synthesis (APS) v2"
    assert certification["islands"] == len(result["certification"]["aleph_points"])
    assert 0.0 <= certification["metrics"]["auc"] <= 1.0
