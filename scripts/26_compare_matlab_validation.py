#!/usr/bin/env python3
"""Compare the threshold rule on the Python plant and the MATLAB reference model.

    python scripts\\26_compare_matlab_validation.py
"""
import argparse, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/matlab_validation")
ap.add_argument("--matlab", default="matlab_metrics.csv")
a = ap.parse_args()

py = pd.read_csv(os.path.join(a.dir, "python_metrics.csv"))
ml = pd.read_csv(os.path.join(a.dir, a.matlab))
m = py.merge(ml, on=["attack", "seed", "arm"], suffixes=("_py", "_ml"))
print(f"{len(m)} paired runs ({m['attack'].nunique()} classes)\n")

def table(df):
    g = df.groupby(["attack", "arm"])
    return pd.DataFrame({
        "f_rms_mHz py": g["f_rms_mHz_py"].mean(), "f_rms_mHz ml": g["f_rms_mHz_ml"].mean(),
        "p_share py": g["p_share_py"].mean(), "p_share ml": g["p_share_ml"].mean(),
        "latency_ms py": g["latency_ms_py"].median(), "latency_ms ml": g["latency_ms_ml"].median(),
        "collapsed py": g["collapsed_py"].sum(), "collapsed ml": g["collapsed_ml"].sum()})

pd.set_option("display.width", 200)
print(table(m).round(3).to_string())

# The claim being validated: the rule's improvement over nominal, per class.
w = m.pivot_table(index=["attack", "seed"], columns="arm",
                  values=["f_rms_mHz_py", "f_rms_mHz_ml", "p_share_py", "p_share_ml"])
red = pd.DataFrame({
    "f reduction py %": 100 * (1 - w[("f_rms_mHz_py", "threshold")] / w[("f_rms_mHz_py", "nominal")]),
    "f reduction ml %": 100 * (1 - w[("f_rms_mHz_ml", "threshold")] / w[("f_rms_mHz_ml", "nominal")]),
    "rule better py": w[("f_rms_mHz_py", "threshold")] <= w[("f_rms_mHz_py", "nominal")],
    "rule better ml": w[("f_rms_mHz_ml", "threshold")] <= w[("f_rms_mHz_ml", "nominal")],
}).replace([np.inf, -np.inf], np.nan)
s = red.groupby(level="attack").agg({"f reduction py %": "median", "f reduction ml %": "median",
                                     "rule better py": "sum", "rule better ml": "sum"})
print("\nFrequency-error reduction by the rule (median over seeds) and seeds where it helps:")
print(s.round(1).to_string())
out = os.path.join(a.dir, "comparison.csv")
table(m).to_csv(out)
print(f"\nwritten: {out}")
