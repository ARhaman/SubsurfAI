"""
PetroMind LAS Reader
Loads and normalises LAS files. Returns a clean DataFrame + metadata.
"""
import lasio
import numpy as np
import pandas as pd
from pathlib import Path

# Common curve name aliases
CURVE_ALIASES = {
    "GR":   ["GR", "GR_ARC", "GRC", "HCGR", "GRS", "ECGR", "GRGC"],
    "NPHI": ["NPHI", "NEU", "TNPH", "CNCF", "NPOR", "NPHI_SAN"],
    "RHOB": ["RHOB", "DEN", "RHOZ", "ZDEN", "RDEN", "RHO8"],
    "RT":   ["RT", "ILD", "LLD", "RLLD", "HDRS", "RD", "AT90", "M2RX"],
    "DT":   ["DT", "AC", "DTCO", "DT4P", "DTTI", "DT24"],
    "CALI": ["CALI", "CAL", "HCAL", "C1", "C2"],
    "SP":   ["SP", "SPS"],
    "MSFL": ["MSFL", "SFLU", "SFL", "RXO", "LLS"],
}


def _find_curve(df: pd.DataFrame, target: str) -> str | None:
    for alias in CURVE_ALIASES.get(target, [target]):
        if alias in df.columns:
            return alias
    return None


def load_las(path) -> tuple[pd.DataFrame, dict]:
    """
    Load a LAS file. Returns (df, meta) where meta has well name, depth range, curves.
    """
    las = lasio.read(str(path))
    df = las.df().reset_index()
    df.columns = [c.upper().strip() for c in df.columns]

    # Standardise depth column name
    depth_col = next((c for c in df.columns if c in ["DEPTH", "DEPT", "MD", "TVD"]), df.columns[0])
    df = df.rename(columns={depth_col: "DEPTH"})

    # Replace bad values with NaN
    df = df.replace([-9999, -999.25, -9999.25, 9999, 1e30], np.nan)
    df = df.dropna(subset=["DEPTH"])

    # Map standard names (first alias found wins)
    col_map = {}
    for std, _ in CURVE_ALIASES.items():
        found = _find_curve(df, std)
        if found and found != std:
            col_map[found] = std
    df = df.rename(columns=col_map)

    # Coalesce resistivity: RT ← best of RT, LLD, LLS, ILD, MSFL by valid-count
    res_candidates = ["RT", "LLD", "LLS", "ILD", "MSFL", "SN", "RD"]
    present = [c for c in res_candidates if c in df.columns]
    if present:
        # Sort by number of valid values descending, fill RT from each in turn
        present_sorted = sorted(present, key=lambda c: df[c].notna().sum(), reverse=True)
        rt_merged = df[present_sorted[0]].copy()
        for c in present_sorted[1:]:
            rt_merged = rt_merged.fillna(df[c])
        df["RT"] = rt_merged

    meta = {
        "well_name":    las.well.WELL.value if hasattr(las.well, "WELL") else Path(path).stem,
        "depth_min":    float(df["DEPTH"].min()),
        "depth_max":    float(df["DEPTH"].max()),
        "curves":       list(df.columns),
        "n_samples":    len(df),
        "raw_las":      las,
    }
    return df, meta


def generate_demo_las(n=500, seed=42) -> tuple[pd.DataFrame, dict]:
    """
    Generate a synthetic LAS dataset for demonstration when no file is uploaded.
    Simulates 4 lithological zones.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(1500, 2000, n)

    # Zone boundaries
    z = np.zeros(n)
    z[(depth >= 1550) & (depth < 1650)] = 1   # sandstone
    z[(depth >= 1700) & (depth < 1780)] = 2   # limestone
    z[(depth >= 1840) & (depth < 1870)] = 3   # coal

    # Log templates per zone
    gr_base   = np.where(z == 0, 85, np.where(z == 1, 35, np.where(z == 2, 20, 18)))
    nphi_base = np.where(z == 0, 0.32, np.where(z == 1, 0.18, np.where(z == 2, 0.08, 0.65)))
    rhob_base = np.where(z == 0, 2.45, np.where(z == 1, 2.38, np.where(z == 2, 2.68, 1.35)))
    rt_base   = np.where(z == 0, 3.0, np.where(z == 1, 25.0, np.where(z == 2, 80.0, 500.0)))
    dt_base   = np.where(z == 0, 105, np.where(z == 1, 75, np.where(z == 2, 50, 140)))

    df = pd.DataFrame({
        "DEPTH": depth,
        "GR":    gr_base   + rng.normal(0, 5, n),
        "NPHI":  nphi_base + rng.normal(0, 0.015, n),
        "RHOB":  rhob_base + rng.normal(0, 0.025, n),
        "RT":    np.clip(rt_base * np.exp(rng.normal(0, 0.3, n)), 0.5, 2000),
        "DT":    dt_base   + rng.normal(0, 3, n),
        "CALI":  8.5 + rng.normal(0, 0.3, n),
        "_ZONE": z,
    })

    meta = {
        "well_name": "DEMO-WELL-A7",
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "curves":    list(df.columns),
        "n_samples": n,
        "is_demo":   True,
    }
    return df, meta
