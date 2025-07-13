#!/usr/bin/env python
"""
train_model.py – Train Gradient Boosting models for thrust & TSFC
-----------------------------------------------------------------
Outputs (to output/):
  • thrust_model.pkl, tsfc_model.pkl
  • residuals_thrust.png, residuals_tsfc.png
  • console metrics (R², RMSE)
"""

import pathlib, joblib, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── paths ───────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parent.parent
CSV    = ROOT / "output" / "clean_engine_perf.csv"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)

# ── load dataset ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

# ---- numeric feature matrix --------------------------------------------
feature_cols = ["mode_pct", "BPR", "OPR", "year"]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# fill NaN column‑wise (no chained assignment)
for col in feature_cols:
    X[col] = X[col].fillna(X[col].mean())

# ---- helper function ----------------------------------------------------
def train_and_report(target, tag):
    y = df[target]
    Xtr, Xts, ytr, yts = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, random_state=42
    )
    model.fit(Xtr, ytr)
    ypred = model.predict(Xts)

    r2   = r2_score(yts, ypred)
    rmse = mean_squared_error(yts, ypred) ** 0.5
    print(f"{tag:<6}  R² = {r2:.3f}   RMSE = {rmse:.4f}")

    # residual histogram
    residuals = yts - ypred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title(f"Residuals – {tag}")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"residuals_{tag}.png", dpi=120)
    plt.close()

    joblib.dump(model, OUTDIR / f"{tag}_model.pkl")

# ---- train both models --------------------------------------------------
train_and_report("thrust_kN", "thrust")
train_and_report("TSFC",      "tsfc")
