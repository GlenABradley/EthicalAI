import json, os, pathlib, subprocess, sys
from ethicalai.api.axes import ACTIVE
from ethicalai.types import AxisPack, Axis
import numpy as np

ART = pathlib.Path("artifacts")
ART.mkdir(exist_ok=True)

def _make_pack(pack_id="cli-test-pack", dim=16):
    # Simple orthonormal axis (single axis) for speed/determinism
    v = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    np.savez_compressed(ART / f"axis_pack:{pack_id}.npz", autonomy=v)
    (ART / f"axis_pack:{pack_id}.meta.json").write_text(json.dumps({"meta": {"note":"test"}}))
    return pack_id

def test_calibrate_cli_runs_and_writes_reports(tmp_path, monkeypatch):
    # Ensure encoder uses fallback for CI (Phase 1 adapter should respect this automatically)
    pack_id = _make_pack()
    reports = tmp_path / "reports"
    cmd = [
        sys.executable, "-m", "ethicalai.axes.calibrate",
        "--pack", pack_id,
        "--dataset", "data/calibration/autonomy.jsonl",
        "--dataset", "data/calibration/truth.jsonl",
        "--reports", str(reports)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    outdir = reports / f"calibration:{pack_id}"
    assert (outdir / "summary.json").exists()
    # Thresholds persisted
    meta = json.loads((ART / f"axis_pack:{pack_id}.meta.json").read_text())
    assert "thresholds" in meta and "autonomy" in meta["thresholds"]
