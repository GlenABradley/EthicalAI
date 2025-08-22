from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

POSTMAN_SCHEMA = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"


def build_url(base_url: str, path: str) -> str:
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    return f"{base_url}{path}"


def to_postman(openapi: Dict[str, Any], base_url: str, name: str = "Coherence API") -> Dict[str, Any]:
    items = []
    paths = openapi.get("paths", {})

    for path, methods in paths.items():
        for method, spec in methods.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                continue
            req_name = spec.get("summary") or spec.get("operationId") or f"{method.upper()} {path}"

            url = build_url(base_url, path)
            headers = [{"key": "Content-Type", "value": "application/json"}]

            body = None
            if "requestBody" in spec:
                body = {
                    "mode": "raw",
                    "raw": json.dumps({}, indent=2),
                    "options": {"raw": {"language": "json"}},
                }

            items.append(
                {
                    "name": req_name,
                    "request": {
                        "method": method.upper(),
                        "header": headers,
                        "url": url,
                        **({"body": body} if body else {}),
                    },
                    "response": [],
                }
            )

    collection = {
        "info": {
            "name": name,
            "schema": POSTMAN_SCHEMA,
            "description": "Auto-generated from OpenAPI",
        },
        "item": items,
    }
    return collection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="Input OpenAPI JSON path")
    ap.add_argument("--out", dest="out", type=Path, required=True, help="Output Postman collection JSON path")
    ap.add_argument("--base-url", dest="base_url", type=str, default="http://localhost:8080", help="Base URL")
    args = ap.parse_args()

    openapi = json.loads(args.inp.read_text(encoding="utf-8"))
    col = to_postman(openapi, base_url=args.base_url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(col, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Postman collection to {args.out}")


if __name__ == "__main__":
    main()
