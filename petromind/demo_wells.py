"""
SubsurfAI — 20 global demo wells across 6 countries / 5 basins.
Each well is a real geological setting with realistic log statistics.
"""
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


DEMO_WELLS = {

    # ── NETHERLANDS (NLOG F-block) ─────────────────────────────────────────────
    "🇳🇱 F03-04 — Dutch NS benchmark (GR/RHOB/NPHI/RT)": {
        "file": "F03-04.las", "lat": 54.852, "lon": 4.833,
        "country": "Netherlands", "basin": "Dutch North Sea",
        "depth_range": "2661–3153 m", "block": "F03",
        "lithology": "Sandstone / Shale",
        "description": "Most cited well in geoscience ML. Rotliegend sandstone reservoir. FORCE benchmark.",
    },
    "🇳🇱 F03-02 — Best Dutch reservoir (φ=19.8%)": {
        "file": "F03-02.las", "lat": 54.836, "lon": 4.802,
        "country": "Netherlands", "basin": "Dutch North Sea",
        "depth_range": "1641–2148 m", "block": "F03",
        "lithology": "Sandstone / Shale",
        "description": "Highest effective porosity in F-block. 1,779 hydrocarbon rows identified.",
    },
    "🇳🇱 F03-01 — Rotliegend gas sand": {
        "file": "F03-01.las", "lat": 54.861, "lon": 4.771,
        "country": "Netherlands", "basin": "Dutch North Sea",
        "depth_range": "2500–3100 m", "block": "F03",
        "lithology": "Sandstone / Shale",
        "description": "Rotliegend aeolian sandstone, Permian gas reservoir.",
    },
    "🇳🇱 F03-03 — Chalk + Rotliegend": {
        "file": "F03-03.las", "lat": 54.844, "lon": 4.817,
        "country": "Netherlands", "basin": "Dutch North Sea",
        "depth_range": "1800–3200 m", "block": "F03",
        "lithology": "Chalk / Sandstone / Shale",
        "description": "Multi-formation well with Chalk and Rotliegend intervals.",
    },
    "🇳🇱 F02-01 — F-block exploration": {
        "file": "F02-01.las", "lat": 54.912, "lon": 4.651,
        "country": "Netherlands", "basin": "Dutch North Sea",
        "depth_range": "2200–3400 m", "block": "F02",
        "lithology": "Sandstone / Shale",
        "description": "F02 block exploration well, Triassic to Rotliegend section.",
    },
    "🇳🇱 F06-01 — Northern F-block": {
        "file": "F06-01.las", "lat": 54.788, "lon": 4.901,
        "country": "Netherlands", "basin": "Dutch North Sea",
        "depth_range": "1900–2900 m", "block": "F06",
        "lithology": "Sandstone / Shale",
        "description": "F06 sub-block, Rotliegend reservoir with gas shows.",
    },

    # ── NORWAY ─────────────────────────────────────────────────────────────────
    "🇳🇴 15/9-19A — Statfjord (Brent Group)": {
        "file": "NO_15-9-19A.las", "lat": 58.4385, "lon": 1.8356,
        "country": "Norway", "basin": "North Sea (Norwegian)",
        "depth_range": "2100–3400 m", "block": "15/9",
        "lithology": "Sandstone / Shale",
        "description": "Statfjord field, Brent Group sandstone. Classic Norwegian oil producer.",
    },
    "🇳🇴 34/10-23 — Oseberg (Brent/Statfjord)": {
        "file": "NO_34-10-23.las", "lat": 60.5041, "lon": 2.7833,
        "country": "Norway", "basin": "North Sea (Norwegian)",
        "depth_range": "1900–3100 m", "block": "34/10",
        "lithology": "Sandstone / Limestone / Shale",
        "description": "Oseberg field. Mixed sandstone-carbonate sequence.",
    },
    "🇳🇴 25/11-1 — Troll (giant gas field)": {
        "file": "NO_25-11-1.las", "lat": 60.6411, "lon": 3.7033,
        "country": "Norway", "basin": "North Sea (Norwegian)",
        "depth_range": "1400–2200 m", "block": "25/11",
        "lithology": "Sandstone / Shale",
        "description": "Troll field — shallow Jurassic sandstone, Europe's largest gas field.",
    },
    "🇳🇴 6507/11-5 — Norne (Åre/Ile Fm)": {
        "file": "NO_6507-11-5.las", "lat": 65.3241, "lon": 8.0833,
        "country": "Norway", "basin": "Norwegian Sea",
        "depth_range": "2500–3800 m", "block": "6507/11",
        "lithology": "Sandstone / Coal / Shale",
        "description": "Norne field. Åre Formation with coal seams. Benchmark dataset.",
    },

    # ── USA ────────────────────────────────────────────────────────────────────
    "🇺🇸 Midland-A12 — Permian Basin (Wolfcamp)": {
        "file": "US_Midland-A12.las", "lat": 31.9974, "lon": -101.9543,
        "country": "USA", "basin": "Permian Basin",
        "depth_range": "1500–4200 m", "block": "Midland",
        "lithology": "Limestone / Dolomite / Shale / Sandstone",
        "description": "Wolfcamp/Spraberry tight oil. World's largest oil-producing basin.",
    },
    "🇺🇸 Anadarko-B7 — Springer/Morrow gas": {
        "file": "US_Anadarko-B7.las", "lat": 35.4676, "lon": -98.6784,
        "country": "USA", "basin": "Anadarko Basin",
        "depth_range": "900–3500 m", "block": "Anadarko",
        "lithology": "Sandstone / Limestone / Shale",
        "description": "Anadarko Basin. Springer-Morrow sandstone gas plays.",
    },
    "🇺🇸 Kansas-C3 — Hugoton carbonate": {
        "file": "US_Kansas-C3.las", "lat": 37.6872, "lon": -100.9241,
        "country": "USA", "basin": "Hugoton Gas Area",
        "depth_range": "400–1200 m", "block": "Kansas",
        "lithology": "Limestone / Dolomite / Anhydrite",
        "description": "Hugoton Gas Area. Chase Group carbonate. SEG-2016 benchmark region.",
    },

    # ── UK NORTH SEA ───────────────────────────────────────────────────────────
    "🇬🇧 29/3-1 — Forties (Paleocene fan)": {
        "file": "UK_29-3-1.las", "lat": 57.7341, "lon": 0.5432,
        "country": "UK", "basin": "UK North Sea",
        "depth_range": "2000–3500 m", "block": "29/3",
        "lithology": "Sandstone / Chalk / Shale",
        "description": "Forties field. Paleocene submarine fan sandstone. UK's largest oil field.",
    },
    "🇬🇧 21/2-4 — Brent (Etive/Rannoch)": {
        "file": "UK_21-2-4.las", "lat": 61.0521, "lon": 1.7234,
        "country": "UK", "basin": "UK North Sea",
        "depth_range": "2800–4000 m", "block": "21/2",
        "lithology": "Sandstone / Coal / Limestone",
        "description": "Brent Group type well. Etive and Rannoch sandstone with coal seams.",
    },
    "🇬🇧 15/25-1 — Magnus (Jurassic turbidite)": {
        "file": "UK_15-25-1.las", "lat": 61.6451, "lon": 1.0123,
        "country": "UK", "basin": "UK North Sea",
        "depth_range": "3200–4500 m", "block": "15/25",
        "lithology": "Sandstone / Shale",
        "description": "Magnus field. Upper Jurassic turbidite sandstone reservoir.",
    },

    # ── AUSTRALIA ──────────────────────────────────────────────────────────────
    "🇦🇺 Carnarvon-1 — NW Shelf (Mungaroo Fm)": {
        "file": "AU_Carnarvon-1.las", "lat": -22.3241, "lon": 114.1234,
        "country": "Australia", "basin": "Carnarvon Basin",
        "depth_range": "2500–4000 m", "block": "Carnarvon",
        "lithology": "Sandstone / Limestone / Shale",
        "description": "Carnarvon Basin NW Shelf. Jurassic Mungaroo Fm gas. Giant LNG province.",
    },
    "🇦🇺 Bonaparte-2 — Plover Fm gas condensate": {
        "file": "AU_Bonaparte-2.las", "lat": -12.5432, "lon": 128.6781,
        "country": "Australia", "basin": "Bonaparte Basin",
        "depth_range": "1800–3200 m", "block": "Bonaparte",
        "lithology": "Sandstone / Limestone / Dolomite",
        "description": "Bonaparte Basin. Triassic Plover Formation gas condensate.",
    },

    # ── MIDDLE EAST ────────────────────────────────────────────────────────────
    "🇸🇦 Ghawar-D1 — Arab-D carbonate (world's largest oil field)": {
        "file": "SA_Ghawar-D1.las", "lat": 25.1234, "lon": 49.3421,
        "country": "Saudi Arabia", "basin": "Arabian Platform",
        "depth_range": "1600–2800 m", "block": "Ghawar",
        "lithology": "Limestone / Dolomite / Anhydrite",
        "description": "Ghawar field. Arab-D carbonate reservoir. World's largest oil field.",
    },
    "🇶🇦 NorthField-E1 — Khuff carbonate (world's largest gas field)": {
        "file": "QA_NorthField-E1.las", "lat": 25.8654, "lon": 51.5321,
        "country": "Qatar", "basin": "Arabian Platform",
        "depth_range": "2800–4000 m", "block": "North Field",
        "lithology": "Limestone / Dolomite / Anhydrite",
        "description": "North Field / South Pars. Khuff carbonate. World's largest gas field.",
    },
}


def get_las_path(well_file: str) -> str:
    return os.path.join(_DATA_DIR, well_file)


def well_map_data():
    """Returns list of dicts for map rendering."""
    rows = []
    for name, info in DEMO_WELLS.items():
        rows.append({
            "name": name,
            "file": info["file"],
            "lat": info["lat"],
            "lon": info["lon"],
            "country": info["country"],
            "basin": info["basin"],
            "lithology": info.get("lithology", ""),
            "depth_range": info.get("depth_range", ""),
            "description": info.get("description", ""),
        })
    return rows


COUNTRY_COLORS = {
    "Netherlands":    "#38BDF8",
    "Norway":         "#34D399",
    "USA":            "#F59E0B",
    "UK":             "#A78BFA",
    "Australia":      "#FB923C",
    "Saudi Arabia":   "#F87171",
    "Qatar":          "#E879F9",
}
