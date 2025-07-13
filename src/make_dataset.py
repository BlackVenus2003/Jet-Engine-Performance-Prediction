#!/usr/bin/env python
"""
make_dataset.py – Clean ICAO engine emissions CSV (Kaggle) → tidy CSV
--------------------------------------------------------------------
Creates output/clean_engine_perf.csv with columns:
  engine, BPR, OPR, year, mode_pct, thrust_kN, fuel_kg_s, TSFC
"""

import pandas as pd, pathlib, numpy as np

# ── paths ───────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "emissions.csv"   # adjust name if needed
OUT_CSV = ROOT / "output" / "clean_engine_perf.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

# ── column mapping (exact) ──────────────────────────────────────────────
cols = {
    "Engine Identification":          "engine",
    "B/P Ratio":                      "BPR",
    "Pressure Ratio":                 "OPR",
    "Initial Test Date":              "test_date",
    "Rated Thrust (kN)":              "F_max",
    "Fuel Flow T/O (kg/sec)":         "FF_TO",
    "Fuel Flow C/O (kg/sec)":         "FF_85",
    "Fuel Flow App (kg/sec)":         "FF_30",
    "Fuel Flow Idle (kg/sec)":        "FF_7",
}

raw = pd.read_csv(RAW_CSV)

# keep & rename only needed columns
df = raw[list(cols.keys())].rename(columns=cols)

# extract year as integer from test_date (YYYY-MM-DD or similar)
df["year"] = pd.to_datetime(df["test_date"], errors="coerce").dt.year
df.drop(columns="test_date", inplace=True)

# melt to long format (one row per engine & power setting)
modes_pct = {"FF_TO":1.00, "FF_85":0.85, "FF_30":0.30, "FF_7":0.07}
long = (
    df.melt(id_vars=["engine","BPR","OPR","year","F_max"],
            value_vars=list(modes_pct.keys()),
            var_name="mode", value_name="fuel_kg_s")
      .dropna(subset=["fuel_kg_s"])
      .assign(mode_pct=lambda d: d["mode"].map(modes_pct),
              thrust_kN=lambda d: d["F_max"] * d["mode_pct"],
              TSFC=lambda d: (d["fuel_kg_s"]*3600)/(d["thrust_kN"]*1000))
      .loc[:, ["engine","BPR","OPR","year","mode_pct","thrust_kN","fuel_kg_s","TSFC"]]
)

long.to_csv(OUT_CSV, index=False)
print(f"✅ Clean dataset saved → {OUT_CSV}  ({len(long):,} rows)")
