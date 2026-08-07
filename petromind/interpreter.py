"""
PetroMind Interpreter
Applies physics equations + ML classification to a well log DataFrame.
Outputs an enriched DataFrame with interpreted columns.
"""
import numpy as np
import pandas as pd
from .knowledge_base import (
    archie_sw, wyllie_porosity, rhob_nphi_porosity,
    vcl_from_gr, uncertainty_porosity, classify_lithology,
    LITHOLOGY_COLORS
)


def interpret(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Full interpretation pipeline. Returns df with added columns.
    """
    cfg = {
        "rw":       0.05,
        "a":        1.0,
        "m":        2.0,
        "n":        2.0,
        "rho_ma":   2.65,
        "rho_fl":   1.0,
        "dt_ma":    55.5,
        "dt_fl":    189.0,
        "gr_clean": 30.0,
        "gr_shale": 120.0,
        **(config or {}),
    }

    out = df.copy()

    # ── Vcl ─────────────────────────────────────────────────────────────────
    if "GR" in out:
        out["VCL"] = vcl_from_gr(out["GR"].values, cfg["gr_clean"], cfg["gr_shale"])
    else:
        out["VCL"] = np.nan

    # ── Porosity ─────────────────────────────────────────────────────────────
    if "RHOB" in out and "NPHI" in out:
        phi_d, phi_avg = rhob_nphi_porosity(
            out["RHOB"].values, out["NPHI"].values, cfg["rho_ma"], cfg["rho_fl"]
        )
        out["PHI_D"]   = phi_d
        out["PHI_AVG"] = phi_avg
    elif "DT" in out:
        out["PHI_D"]   = wyllie_porosity(out["DT"].values, cfg["dt_ma"], cfg["dt_fl"])
        out["PHI_AVG"] = out["PHI_D"]
    else:
        out["PHI_D"]   = np.nan
        out["PHI_AVG"] = np.nan

    # Clay correction
    if not out["VCL"].isna().all() and not out["PHI_AVG"].isna().all():
        out["PHI_EFF"] = np.clip(out["PHI_AVG"] - 0.4 * out["VCL"], 0, 0.45)
    else:
        out["PHI_EFF"] = out["PHI_AVG"]

    # ── Water Saturation ─────────────────────────────────────────────────────
    if "RT" in out and not out["PHI_EFF"].isna().all():
        out["SW"] = archie_sw(
            out["RT"].values, cfg["rw"],
            np.where(out["PHI_EFF"].values > 0.02, out["PHI_EFF"].values, 0.02),
            cfg["a"], cfg["m"], cfg["n"]
        )
    else:
        out["SW"] = np.nan

    # ── Uncertainty ───────────────────────────────────────────────────────────
    if not out["VCL"].isna().all():
        out["UNC_PHI"] = uncertainty_porosity(out["PHI_EFF"].values, out["VCL"].values)
        out["UNC_SW"]  = np.where(out["SW"].notna(), 0.05 + out["VCL"] * 0.10, np.nan)
    else:
        out["UNC_PHI"] = np.nan
        out["UNC_SW"]  = np.nan

    # ── Lithology ─────────────────────────────────────────────────────────────
    gr   = out.get("GR",   pd.Series(np.full(len(out), 80))).fillna(80)
    nphi = out.get("NPHI", pd.Series(np.full(len(out), 0.25))).fillna(0.25)
    rhob = out.get("RHOB", pd.Series(np.full(len(out), 2.5))).fillna(2.5)

    liths, confs = [], []
    for g, n, r in zip(gr, nphi, rhob):
        lith, conf = classify_lithology(float(g), float(n), float(r))
        liths.append(lith)
        confs.append(conf)
    out["LITHOLOGY"]      = liths
    out["LITH_CONF"]      = confs
    out["LITH_COLOR"]     = [LITHOLOGY_COLORS.get(l, "#DDDDDD") for l in liths]

    # ── Fluid Flag ────────────────────────────────────────────────────────────
    def fluid_flag(sw, phi, vcl):
        if pd.isna(sw) or pd.isna(phi): return "undetermined"
        if vcl > 0.5: return "shale"
        if phi < 0.05: return "tight"
        if sw < 0.40: return "hydrocarbon"
        if sw < 0.65: return "transition"
        return "water"

    out["FLUID_FLAG"] = [
        fluid_flag(sw, phi, vcl)
        for sw, phi, vcl in zip(out.get("SW", [np.nan]*len(out)),
                                out.get("PHI_EFF", [np.nan]*len(out)),
                                out.get("VCL", [0.0]*len(out)))
    ]

    return out


def zone_summary(df: pd.DataFrame, depth_top: float, depth_bot: float) -> dict:
    """Return average properties for a depth interval."""
    mask = (df["DEPTH"] >= depth_top) & (df["DEPTH"] <= depth_bot)
    sub = df[mask]
    if len(sub) == 0:
        return {}

    lith_mode = sub["LITHOLOGY"].mode()[0] if "LITHOLOGY" in sub else "unknown"
    return {
        "n_points":    len(sub),
        "gr":          sub["GR"].mean()    if "GR"      in sub else np.nan,
        "nphi":        sub["NPHI"].mean()  if "NPHI"    in sub else np.nan,
        "rhob":        sub["RHOB"].mean()  if "RHOB"    in sub else np.nan,
        "rt":          sub["RT"].mean()    if "RT"      in sub else np.nan,
        "vcl":         sub["VCL"].mean()   if "VCL"     in sub else np.nan,
        "phi_avg":     sub["PHI_AVG"].mean()if "PHI_AVG" in sub else np.nan,
        "phi_eff":     sub["PHI_EFF"].mean()if "PHI_EFF" in sub else np.nan,
        "sw":          sub["SW"].mean()    if "SW"      in sub else np.nan,
        "unc_phi":     sub["UNC_PHI"].mean()if "UNC_PHI" in sub else np.nan,
        "unc_sw":      sub["UNC_SW"].mean() if "UNC_SW"  in sub else np.nan,
        "lithology":   lith_mode,
        "lith_conf":   sub["LITH_CONF"].mean() if "LITH_CONF" in sub else 0.0,
    }
