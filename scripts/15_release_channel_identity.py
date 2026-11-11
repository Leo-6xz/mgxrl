"""
15_release_channel_identity.py

Are the release's ten DGs ten distinct nodes, or one node reported ten times?

The Tier B per-DG differential raised the question. Under Additive, seven of the
nine unattacked DGs move by 3.904545 W to seven significant figures, while the
plant's ring produces a distance-ordered response in symmetric pairs
(2,10), (3,9), (4,8), (5,7), 6. A ring cannot produce identical responses at
different distances from the attacked node. Either the release's electrical
topology is a common bus, or its per-DG columns are not independent signals.

This distinguishes the two by looking at the columns directly, with no model on
either side. For every scenario it clusters P_DG*, Q_DG* and f_DG* into groups
of columns that are EXACTLY equal over the whole record, and reports the
largest pairwise difference inside each group and between groups.

Reading it:

  exact duplicate columns   The release reports fewer independent DGs than it
                            has columns. Per-DG features in the detector are
                            then partly redundant, and no per-node fidelity
                            claim can be made against it in either direction.

  distinct but near-equal   The DGs are separate nodes on a stiff/common bus.
                            Localisation is physically absent rather than
                            artificial, and Tier C can validate aggregate
                            response only.

  distinct and spread       The uniformity is confined to the attack window and
                            something else explains it. The spread was reported.

Usage (PowerShell, one line):

  python scripts\\15_release_channel_identity.py --dataset C:\\research\\mgxrl\\Dataset
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

N_DG = 10


def find_csvs(root: str) -> dict[str, str]:
    paths = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not paths:
        sys.exit(f"no CSVs under {root}")
    out = {}
    for p in paths:
        name = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        out.setdefault(name, p)
    return out


def cluster(cols: list[str], data: dict[str, np.ndarray]) -> list[list[str]]:
    """Group columns that are exactly equal wherever both are finite."""
    groups: list[list[str]] = []
    for c in cols:
        placed = False
        for g in groups:
            a, b = data[c], data[g[0]]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() and np.array_equal(a[ok], b[ok]):
                g.append(c)
                placed = True
                break
        if not placed:
            groups.append([c])
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"C:\research\mgxrl\Dataset")
    ap.add_argument("--out", default="results/release_channel_identity.csv")
    a = ap.parse_args()

    rows = []
    for scen, path in find_csvs(a.dataset).items():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        cmap = {c.lower(): c for c in df.columns}

        for ch in ("P", "Q", "f"):
            cols = [f"{ch}_DG{i}" for i in range(1, N_DG + 1)]
            have = [c for c in cols if c.lower() in cmap]
            if len(have) < 2:
                continue
            data = {c: pd.to_numeric(df[cmap[c.lower()]],
                                     errors="coerce").to_numpy(dtype=float)
                    for c in have}

            groups = cluster(have, data)
            # spread between the group representatives, on finite samples only
            reps = [g[0] for g in groups]
            spread = 0.0
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    x, y = data[reps[i]], data[reps[j]]
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum():
                        spread = max(spread, float(np.max(np.abs(x[ok] - y[ok]))))

            sizes = sorted((len(g) for g in groups), reverse=True)
            rows.append({
                "scenario": scen, "channel": ch,
                "n_columns": len(have),
                "n_distinct": len(groups),
                "largest_identical_group": sizes[0],
                "group_sizes": "+".join(str(s) for s in sizes),
                "max_between_group_diff": spread,
                "members_of_largest": ",".join(
                    sorted(max(groups, key=len),
                           key=lambda c: int(c.split("DG")[1]))),
            })

    if not rows:
        sys.exit("no DG columns found")

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.to_csv(a.out, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")
    print(res.drop(columns=["members_of_largest"]).to_string(index=False))
    print("\nmembers of the largest identical group, per scenario/channel:")
    for _, r in res.iterrows():
        if r["largest_identical_group"] > 1:
            print(f"  {r['scenario']:>9} {r['channel']}: {r['members_of_largest']}")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
