from __future__ import annotations

from typing import Dict, Iterator, List, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path("data/indices")
BASE.mkdir(parents=True, exist_ok=True)


def _path(axis_pack_id: str, name: str) -> Path:
    d = BASE / axis_pack_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.parquet"


def write_spans(axis_pack_id: str, doc_id: str, records: List[dict]) -> None:
    path = _path(axis_pack_id, "spans")
    table = pa.Table.from_pylist(records)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table], promote=True)
    pq.write_table(table, path)


def iterate_spans(axis_pack_id: str, filters: Optional[dict] = None) -> Iterator[dict]:
    path = _path(axis_pack_id, "spans")
    if not path.exists():
        return iter(())
    table = pq.read_table(path)
    cols = table.to_pydict()
    n = len(next(iter(cols.values()))) if cols else 0
    for i in range(n):
        rec = {k: v[i] for k, v in cols.items()}
        if filters:
            ok = True
            if "minC" in filters and rec.get("C", 0.0) < float(filters["minC"]):
                ok = False
            if ok:
                yield rec
        else:
            yield rec


def write_frames(axis_pack_id: str, doc_id: str, records: List[dict]) -> None:
    path = _path(axis_pack_id, "frames")
    table = pa.Table.from_pylist(records)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table], promote=True)
    pq.write_table(table, path)


def iterate_frames(axis_pack_id: str, filters: Optional[dict] = None) -> Iterator[dict]:
    path = _path(axis_pack_id, "frames")
    if not path.exists():
        return iter(())
    table = pq.read_table(path)
    cols = table.to_pydict()
    n = len(next(iter(cols.values()))) if cols else 0
    for i in range(n):
        rec = {k: v[i] for k, v in cols.items()}
        yield rec
