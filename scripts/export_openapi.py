import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is on sys.path when executing as a script
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # repo root
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def main():
    parser = argparse.ArgumentParser(description="Export FastAPI OpenAPI JSON")
    parser.add_argument("-o", "--out", dest="out", default="docs/openapi.json", help="Output JSON path")
    args = parser.parse_args()

    # Import your existing FastAPI app (do not start the server)
    from coherence.api.main import app

    spec = app.openapi()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"OpenAPI exported to {out_path}")

if __name__ == "__main__":
    main()
