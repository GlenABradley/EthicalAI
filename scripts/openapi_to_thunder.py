from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def build_url(base_url: str, path: str) -> str:
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    return f"{base_url}{path}"


def to_thunder(openapi: Dict[str, Any], base_url: str, name: str = "Coherence API") -> Dict[str, Any]:
    col_id = str(uuid.uuid4())
    requests: List[Dict[str, Any]] = []

    for path, methods in openapi.get("paths", {}).items():
        for method, spec in methods.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                continue
            req_name = spec.get("summary") or spec.get("operationId") or f"{method.upper()} {path}"
            url = build_url(base_url, path)

            body = None
            if "requestBody" in spec:
                body = {"type": "json", "raw": json.dumps({}, indent=2)}

            requests.append(
                {
                    "_id": str(uuid.uuid4()),
                    "colId": col_id,
                    "name": req_name,
                    "url": url,
                    "method": method.upper(),
                    "headers": [{"name": "Content-Type", "value": "application/json"}],
                    **({"body": body} if body else {}),
                }
            )

    thunder = {
        "client": "Thunder Client",
        "collectionName": name,
        "dateExported": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "folders": [],
        "requests": requests,
    }
    return thunder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="Input OpenAPI JSON path")
    ap.add_argument("--out", dest="out", type=Path, required=True, help="Output Thunder Client collection JSON path")
    ap.add_argument("--base-url", dest="base_url", type=str, default="http://localhost:8080", help="Base URL")
    args = ap.parse_args()

    openapi = json.loads(args.inp.read_text(encoding="utf-8"))
    col = to_thunder(openapi, base_url=args.base_url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(col, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Thunder Client collection to {args.out}")


if __name__ == "__main__":
    main()
