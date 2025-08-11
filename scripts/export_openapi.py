from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure coherence/src is on sys.path when executing as a script or module
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # repo root
SRC_PATH = PROJECT_ROOT / "coherence" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from coherence.api.main import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path for OpenAPI schema")
    args = ap.parse_args()

    app = create_app()
    schema = app.openapi()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote OpenAPI to {args.out}")


if __name__ == "__main__":
    main()
