#!/usr/bin/env python3
"""
Renames the arm-position folders of every character, using the table made by
make_release.py --make-poses-template (columns: folder, name, ...).

    <root>/<character>/<folder>   ->   <root>/<character>/<name>

By default it only SHOWS what it would do. Add --apply to rename for real.
Afterwards the table is updated (folder = new name), so it still works with
make_release.py --poses.

Usage:
  python3 rename_poses.py esp32 poses.csv            # preview
  python3 rename_poses.py esp32 poses.csv --apply    # rename
"""

import argparse
import csv
import os
import re
import sys


def std_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser(description="Rename arm-position folders in every character folder.")
    ap.add_argument("root", help="photo folder: <root>/<character>/<arm_position>/")
    ap.add_argument("table", help="poses.csv with the columns folder and name")
    ap.add_argument("--apply", action="store_true", help="really rename (default: only show)")
    a = ap.parse_args()

    with open(a.table, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "folder" not in rows[0] or "name" not in rows[0]:
        sys.exit(f"{a.table}: expected the columns folder and name")
    mapping = {}
    for r in rows:
        old, new = (r["folder"] or "").strip(), std_name(r["name"] or "")
        if not old or not new:
            sys.exit(f"{a.table}: empty folder or name in row {r}")
        mapping[old] = new
    if len(set(mapping.values())) != len(mapping):
        sys.exit(f"{a.table}: two folders would get the same name")

    characters = sorted(d for d in os.listdir(a.root)
                        if os.path.isdir(os.path.join(a.root, d)) and not d.startswith((".", "_")))
    moves, problems = [], []
    for c in characters:
        present = {d for d in os.listdir(os.path.join(a.root, c)) if os.path.isdir(os.path.join(a.root, c, d))}
        for old, new in mapping.items():
            if old == new:
                continue
            if old not in present:
                if new not in present:
                    problems.append(f"{c}: no folder '{old}'")
                continue                                   # already renamed earlier
            if new in present:
                problems.append(f"{c}: '{new}' already exists, can't rename '{old}'")
                continue
            moves.append((os.path.join(a.root, c, old), os.path.join(a.root, c, new)))
        unknown = present - set(mapping) - set(mapping.values())
        for u in sorted(unknown):
            problems.append(f"{c}: folder '{u}' is not in the table (left as it is)")

    for src, dst in moves:
        print(f"  {os.path.relpath(src, a.root)}  ->  {os.path.basename(dst)}")
    for p in problems:
        print("  WARNING: " + p)
    if not moves:
        print("Nothing to rename.")
        return
    if not a.apply:
        print(f"\n{len(moves)} folders would be renamed. Run again with --apply to do it.")
        return
    if any("already exists" in p for p in problems):
        sys.exit("\nStopped: fix the warnings above first (nothing was renamed).")

    for src, dst in moves:
        os.rename(src, dst)
    for r in rows:                                         # the table now refers to the new names
        r["name"] = std_name(r["name"])
        r["folder"] = r["name"]
    with open(a.table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nRenamed {len(moves)} folders. {a.table} updated (folder = new name).")


if __name__ == "__main__":
    main()
