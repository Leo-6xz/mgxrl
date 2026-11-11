#!/usr/bin/env python3
"""Locate malformed values in the raw release CSVs.

This script finds the offending physical LINES and prints them, so you can see
the actual text rather than guess from a summary statistic.

USAGE
    python scripts/11_check_csv_integrity.py --root /path/to/dataport
    python scripts/11_check_csv_integrity.py --root ... --scenario Normal --show 20
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

PLAUSIBLE_MAX = 1e7          # just a ceiling for volts/amps/watts/Hz


def scan_file(path, show=10, plausible_max=PLAUSIBLE_MAX):
    print(f"\n{'=' * 72}\n{os.path.basename(path)}\n{'=' * 72}")

    # --- pass 0: field-count check FIRST -----------------------------------
    with open(path, "r", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        header_n = len(header)
        counts, ragged_lines = {}, []
        for lineno, line in enumerate(fh, start=2):
            k = len(line.rstrip("\n").split(","))
            counts[k] = counts.get(k, 0) + 1
            if k != header_n and len(ragged_lines) < show:
                ragged_lines.append((lineno, k, line.rstrip("\n")))

    print(f"  header fields : {header_n}")
    print(f"  row field counts: {counts}")
    if len(counts) > 1 or (counts and header_n not in counts):
        print("\n  RAGGED ROWS -- fields are shifted on some lines, so columns")
        print("  hold each other's values there. This alone explains impossible")
        print("  magnitudes and NaN in a single column.")
        for lineno, k, line in ragged_lines:
            cells = line.split(",")
            odd = [f"col {j}: {c!r}" for j, c in enumerate(cells) if _suspicious(c)]
            print(f"    line {lineno}: {k} fields (expected {header_n})")
            for o in odd[:3]:
                print(f"        {o}")
        return {-1}

    # --- pass 1: parse and locate suspicious cells -------------------------
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  pandas could not parse this file: {e}")
        return {-1}
    num = df.apply(pd.to_numeric, errors="coerce")

    bad_rows = set()
    report = []
    for c in num.columns:
        col = num[c]
        n_nan = int(col.isna().sum())
        n_inf = int(np.isinf(col.to_numpy(dtype="float64", na_value=0.0)).sum())
        finite = col.to_numpy(dtype="float64", na_value=0.0)
        n_huge = int((np.abs(finite) > plausible_max).sum())
        if n_nan or n_inf or n_huge:
            report.append((c, n_nan, n_inf, n_huge, float(np.nanmax(np.abs(col)))))
            idx = col.index[col.isna() | (col.abs() > plausible_max)
                            | np.isinf(col.fillna(0))]
            bad_rows.update(idx.tolist()[:200])

    if not report:
        print("  clean: all values parse as finite numbers within range")
        return set()

    print(f"  {'column':<12}{'NaN':>8}{'inf':>8}{'|x|>1e7':>10}{'max |x|':>16}")
    for c, n_nan, n_inf, n_huge, mx in report:
        print(f"  {c:<12}{n_nan:>8,}{n_inf:>8,}{n_huge:>10,}{mx:>16.4g}")

    # --- pass 2: show the raw TEXT of the offending lines -------------------
    print(f"\n  raw text of up to {show} offending lines "
          f"(line 1 = header, so CSV row i is line i+2):")
    want = sorted(bad_rows)[:show]
    want_lines = {i + 2 for i in want}
    with open(path, "r", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno in want_lines:
                cells = line.rstrip("\n").split(",")
                flagged = [
                    f"[col {j}: {cell!r}]" for j, cell in enumerate(cells)
                    if cell.strip() == ""
                    or _suspicious(cell)
                ]
                print(f"    line {lineno}: {len(cells)} fields"
                      + ("  " + " ".join(flagged[:4]) if flagged else
                         "  (field count/format looks normal -- inspect manually)"))
            if lineno > max(want_lines, default=0):
                break

    return bad_rows


def _suspicious(cell: str) -> bool:
    c = cell.strip()
    if c == "":
        return True
    try:
        v = float(c)
    except ValueError:
        return True
    return not np.isfinite(v) or abs(v) > PLAUSIBLE_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario", default=None, help="default: all")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()

    from mgxrl.detection.dataio import SCENARIOS
    scenarios = [a.scenario] if a.scenario else SCENARIOS

    any_bad = False
    for s in scenarios:
        for path in sorted(glob.glob(os.path.join(a.root, s, "*.csv"))):
            if scan_file(path, a.show):
                any_bad = True

    print(f"\n{'=' * 72}")
    if any_bad:
        print("Malformed values found. Until these are understood, I think it is wise to treat with caution")
        print()
        print("If the cause is ragged rows or a lost separator, re-export from")
        print("MATLAB with writematrix(...,'Delimiter',',') at full precision rather")
        print("than via Excel, which reformats and truncates.")
    else:
        print("All files parse cleanly.")


if __name__ == "__main__":
    main()
