from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class FrameStore:
    """SQLite-backed frame store using a simple schema.

    Tables:
      - frames(frame_id PK, doc_id, pack_id, pack_hash, k, d, predicate_start, predicate_end, roles_json, meta_json, created_at)
      - frame_axis(frame_id, axis_idx, coord) PRIMARY KEY(frame_id, axis_idx)
      - frame_vectors(frame_id PK, vec BLOB)
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Persistent connection for router lifetime
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # PRAGMAs
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.commit()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        # 1) Create tables if missing
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS frames (
              frame_id TEXT PRIMARY KEY,
              doc_id TEXT,
              pack_id TEXT,
              pack_hash TEXT,
              k INTEGER,
              d INTEGER,
              predicate_start INTEGER,
              predicate_end INTEGER,
              roles_json TEXT,
              meta_json TEXT,
              created_at INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_axis (
              frame_id TEXT,
              axis_idx INTEGER,
              coord REAL,
              PRIMARY KEY(frame_id, axis_idx)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_vectors (
              frame_id TEXT PRIMARY KEY,
              vec BLOB
            )
            """
        )
        self.conn.commit()

        # 2) Migrate missing columns, then backfill
        cur.execute("PRAGMA table_info(frames)")
        cols = [r[1] for r in cur.fetchall()]
        if "created_at" not in cols:
            cur.execute("ALTER TABLE frames ADD COLUMN created_at INTEGER;")
            cur.execute("UPDATE frames SET created_at = CAST(strftime('%s','now') AS INTEGER) WHERE created_at IS NULL;")
            self.conn.commit()

        # 3) Create indexes after column exists
        cur.execute("CREATE INDEX IF NOT EXISTS ix_frames_doc_id ON frames(doc_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_frames_created_at ON frames(created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_axis_idx_coord ON frame_axis(axis_idx, coord);")
        self.conn.commit()

    @staticmethod
    def _to_blob(vec: Iterable[float]) -> bytes:
        arr = np.asarray(list(vec), dtype=np.float32)
        return arr.astype("<f4").tobytes()

    def put(
        self,
        *,
        doc_id: str,
        frames: List[Dict[str, Any]],
        frame_vectors: Optional[List[List[float]]],
        pack_id: str,
        pack_hash: str,
        k: int,
        d: int,
    ) -> int:
        """Upsert frames and axis coords; vectors optional.

        Each element in frames supports keys:
         - id (required)
         - predicate [start,end]
         - roles: {role: [s,e]}
         - coords: [k]
         - role_coords: {role: [k]}
         - meta: {...}
        """
        cur = self.conn.cursor()
        count = 0
        for i, f in enumerate(frames):
                fid = f["id"]
                pred = f.get("predicate") or [0, 0]
                roles = f.get("roles") or {}
                meta = f.get("meta") or {}
                coords = f.get("coords")
                role_coords = f.get("role_coords") or {}
                # Validate lengths if provided
                if coords is not None and len(coords) != k:
                    raise ValueError("coords length must equal k")
                for rname, rvec in role_coords.items():
                    if len(rvec) != k:
                        raise ValueError("role coords length must equal k")
                created_at = int(time.time())
                # Upsert frame row
                cur.execute(
                    """
                    INSERT INTO frames(frame_id, doc_id, pack_id, pack_hash, k, d, predicate_start, predicate_end, roles_json, meta_json, created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(frame_id) DO UPDATE SET
                      doc_id=excluded.doc_id,
                      pack_id=excluded.pack_id,
                      pack_hash=excluded.pack_hash,
                      k=excluded.k,
                      d=excluded.d,
                      predicate_start=excluded.predicate_start,
                      predicate_end=excluded.predicate_end,
                      roles_json=excluded.roles_json,
                      meta_json=excluded.meta_json,
                      created_at=COALESCE(frames.created_at, excluded.created_at)
                    """,
                    (
                        fid,
                        doc_id,
                        pack_id,
                        pack_hash,
                        int(k),
                        int(d),
                        int(pred[0]),
                        int(pred[1]),
                        json.dumps({"roles": roles, "role_coords": role_coords}),
                        json.dumps(meta),
                        created_at,
                    ),
                )
                # Replace axis coords
                cur.execute("DELETE FROM frame_axis WHERE frame_id=?", (fid,))
                if coords is not None:
                    cur.executemany(
                        "INSERT INTO frame_axis(frame_id, axis_idx, coord) VALUES(?,?,?)",
                        ((fid, idx, float(val)) for idx, val in enumerate(coords)),
                    )
                # Replace vector blob if provided
                if frame_vectors and i < len(frame_vectors) and frame_vectors[i] is not None:
                    vec = frame_vectors[i]
                    if len(vec) != 3 * d:
                        raise ValueError("frame vector length must equal 3*d")
                    cur.execute("DELETE FROM frame_vectors WHERE frame_id=?", (fid,))
                    cur.execute(
                        "INSERT INTO frame_vectors(frame_id, vec) VALUES(?,?)",
                        (fid, self._to_blob(vec)),
                    )
                count += 1
        self.conn.commit()
        return count

    def search(self, *, axis_idx: int, min_val: float, max_val: float, limit: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
                """
                SELECT f.frame_id, f.doc_id, f.pack_id, f.pack_hash, f.k, f.d,
                       f.predicate_start, f.predicate_end, fa.coord
                FROM frame_axis fa
                JOIN frames f ON f.frame_id = fa.frame_id
                WHERE fa.axis_idx=? AND fa.coord BETWEEN ? AND ?
                LIMIT ?
                """,
                (int(axis_idx), float(min_val), float(max_val), int(limit)),
            )
        rows = cur.fetchall()
        items: List[Dict[str, Any]] = []
        for r in rows:
                items.append(
                    {
                        "frame_id": r[0],
                        "doc_id": r[1],
                        "pack_id": r[2],
                        "pack_hash": r[3],
                        "k": r[4],
                        "d": r[5],
                        "predicate": [r[6], r[7]],
                        "coord": r[8],
                        "axis_idx": axis_idx,
                    }
                )
        return items

    def trace(self, *, entity_str: str, limit: int) -> List[Dict[str, Any]]:
        # MVP: search meta_json text for case-insensitive match
        term = entity_str.lower()
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT frame_id, doc_id, pack_id, pack_hash, k, d, predicate_start, predicate_end, meta_json
            FROM frames
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        items: List[Dict[str, Any]] = []
        for r in rows:
            meta_text = r[8] or ""
            if term in meta_text.lower():
                items.append(
                    {
                        "frame_id": r[0],
                        "doc_id": r[1],
                        "pack_id": r[2],
                        "pack_hash": r[3],
                        "k": r[4],
                        "d": r[5],
                        "predicate": [r[6], r[7]],
                    }
                )
        return items


# Factory to create a store at a given db path
def create_store(db_path: Path) -> FrameStore:
    return FrameStore(db_path)
