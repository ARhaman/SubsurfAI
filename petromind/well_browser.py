"""
Well browser — searchable catalogue of all 1,066 NLOG Dutch North Sea wells.
"""
import json, os
import pandas as pd

_CAT_PATH = os.path.join(os.path.dirname(__file__), "well_catalogue.json")
# LAS folder: same dir as SubsurfAI, go up one level
_LAS_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_catalogue() -> pd.DataFrame:
    with open(_CAT_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["depth_range_m"] = (df["depth_max"] - df["depth_min"]).round(0)
    df["suite"] = df.apply(lambda r: (
        ("GR " if r["has_gr"] else "") +
        ("RHOB " if r["has_rhob"] else "") +
        ("NPHI " if r["has_nphi"] else "") +
        ("RT " if r["has_rt"] else "") +
        ("DT" if r["has_dt"] else "")
    ).strip(), axis=1)
    return df


def get_las_path(well_file: str) -> str:
    """Return absolute path to a LAS file by filename."""
    return os.path.join(_LAS_DIR, well_file)


def filter_wells(df: pd.DataFrame, block=None, needs_full_suite=False,
                 needs_gr=False, needs_rhob=False, needs_rt=False,
                 min_depth=None, max_depth=None, search=None) -> pd.DataFrame:
    out = df.copy()
    if block and block != "All":
        out = out[out["block"] == block]
    if needs_full_suite:
        out = out[out["full_suite"]]
    if needs_gr:
        out = out[out["has_gr"]]
    if needs_rhob:
        out = out[out["has_rhob"]]
    if needs_rt:
        out = out[out["has_rt"]]
    if min_depth is not None:
        out = out[out["depth_max"] >= min_depth]
    if max_depth is not None:
        out = out[out["depth_min"] <= max_depth]
    if search:
        out = out[out["well"].str.upper().str.contains(search.upper())]
    return out.sort_values("well").reset_index(drop=True)
