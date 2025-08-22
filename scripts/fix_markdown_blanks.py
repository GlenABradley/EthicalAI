from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

LIST_MARKERS = ("- ", "* ")


def is_heading(line: str) -> bool:
    return line.lstrip().startswith("#") and not line.lstrip().startswith("#######!")


def is_fence(line: str) -> bool:
    s = line.strip()
    return s.startswith("```") or s.startswith("~~~")


def is_list_item(line: str) -> bool:
    ls = line.lstrip()
    if any(ls.startswith(m) for m in LIST_MARKERS):
        return True
    # ordered list: 1. foo
    i = 0
    while i < len(ls) and ls[i].isdigit():
        i += 1
    return i > 0 and i + 1 <= len(ls) and ls[i:i+2] == ". "


def ensure_blank(lines: List[str], idx: int, before: bool) -> None:
    if before:
        if idx > 0 and lines[idx-1].strip() != "":
            lines.insert(idx, "\n")
    else:
        if idx + 1 < len(lines) and lines[idx+1].strip() != "":
            lines.insert(idx+1, "\n")


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = original[:]

    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if is_fence(line):
            # surround fences with blank lines
            ensure_blank(lines, i, before=True)
            # find matching closing fence
            i += 1
            in_fence = True
            while i < len(lines) and not is_fence(lines[i]):
                i += 1
            if i < len(lines):
                # closing fence at i
                ensure_blank(lines, i, before=False)
            in_fence = False
        elif is_heading(line):
            # ensure blank line before and after heading
            ensure_blank(lines, i, before=True)
            ensure_blank(lines, i, before=False)
        elif is_list_item(line):
            # ensure blank line before list block start
            if i == 0 or lines[i-1].strip() != "":
                # if previous line is not blank and not a list item, insert blank
                if i == 0 or not is_list_item(lines[i-1]):
                    lines.insert(i, "\n")
                    i += 1
            # ensure blank after the end of a list block
            j = i
            while j < len(lines) and (is_list_item(lines[j]) or lines[j].strip() == ""):
                j += 1
            # j is first non-list and non-blank after block
            if j <= len(lines) - 1 and lines[j-1].strip() != "":
                lines.insert(j, "\n")
            i = j
            continue
        i += 1

    changed = lines != original
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Fix markdown blanks around headings, lists, and fences")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    any_changed = False
    for p in args.files:
        if p.exists() and p.suffix.lower() in {".md", ".markdown"}:
            if process_file(p):
                print(f"Fixed: {p}")
                any_changed = True
        else:
            print(f"Skip (not found or not markdown): {p}")
    if not any_changed:
        print("No changes needed.")


if __name__ == "__main__":
    main()
