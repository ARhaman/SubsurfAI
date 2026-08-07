"""
Demo well catalogue — real NLOG Dutch North Sea F-block wells.
Coordinates sourced from Dutch Continental Shelf block grid (RD/WGS84).
"""
import os

# Absolute path to bundled data folder
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DEMO_WELLS = {
    "F03-04 (F3 benchmark — GR/RHOB/NPHI/ILD)": {
        "file": "F03-04.las",
        "lat": 54.852, "lon": 4.833,
        "depth_range": "2661–3153 m",
        "valid_rows": 4906,
        "block": "F03",
        "description": "Most cited well in geoscience ML (FaciesSAM, GEM 3D). Full suite: GR, RHOB, NPHI, ILD, LLD, DT, CALI, SP.",
        "nlog_url": "https://www.nlog.nl/datacenter/brh-overview?q=F03-04",
    },
    "F03-02 (1641–2147 m, 5 curves)": {
        "file": "F03-02.las",
        "lat": 54.867, "lon": 4.802,
        "depth_range": "1641–2148 m",
        "valid_rows": 5059,
        "block": "F03",
        "description": "Shallow F03 well. GR, RHOB, NPHI, ILD, LLD, DT, SP.",
        "nlog_url": "https://www.nlog.nl/datacenter/brh-overview?q=F03-02",
    },
    "F03-01 (1749–2327 m, 5 curves)": {
        "file": "F03-01.las",
        "lat": 54.843, "lon": 4.793,
        "depth_range": "1749–2327 m",
        "valid_rows": 5773,
        "block": "F03",
        "description": "First F03 borehole. GR, RHOB, NPHI, ILD, DT, SP.",
        "nlog_url": "https://www.nlog.nl/datacenter/brh-overview?q=F03-01",
    },
    "F03-03 (2535–3910 m, deep section)": {
        "file": "F03-03.las",
        "lat": 54.839, "lon": 4.820,
        "depth_range": "2535–3910 m",
        "valid_rows": 9986,
        "block": "F03",
        "description": "Deep F03 well with carbonate section. GR, RHOB, NPHI, ILD, LLD, DT, CALI, SP.",
        "nlog_url": "https://www.nlog.nl/datacenter/brh-overview?q=F03-03",
    },
    "F02-01 (1650–3124 m, 4 curves)": {
        "file": "F02-01.las",
        "lat": 54.887, "lon": 4.668,
        "depth_range": "1650–3124 m",
        "valid_rows": 7361,
        "block": "F02",
        "description": "F02 block well west of F3. GR, RHOB, NPHI, DT.",
        "nlog_url": "https://www.nlog.nl/datacenter/brh-overview?q=F02-01",
    },
    "F06-01 (1836–3526 m, largest dataset)": {
        "file": "F06-01.las",
        "lat": 55.018, "lon": 5.163,
        "depth_range": "1836–3526 m",
        "valid_rows": 16763,
        "block": "F06",
        "description": "Largest dataset — 16,763 valid rows. GR, RHOB, NPHI, DT. Northeast F-block.",
        "nlog_url": "https://www.nlog.nl/datacenter/brh-overview?q=F06-01",
    },
}


def get_las_path(well_name: str) -> str:
    info = DEMO_WELLS[well_name]
    return os.path.join(_DATA_DIR, info["file"])


def well_map_data() -> list[dict]:
    """Return list of dicts for map plotting."""
    rows = []
    for name, info in DEMO_WELLS.items():
        rows.append({
            "name": name.split(" (")[0],
            "label": name,
            "lat": info["lat"],
            "lon": info["lon"],
            "block": info["block"],
            "depth_range": info["depth_range"],
            "valid_rows": info["valid_rows"],
            "description": info["description"],
            "nlog_url": info["nlog_url"],
        })
    return rows
