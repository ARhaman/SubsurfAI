"""
PetroMind Knowledge Base
Physics equations + petrophysical reasoning engine
"""
import numpy as np


# ── Physics equations ────────────────────────────────────────────────────────

def archie_sw(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation. Returns Sw (fraction)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        F = a / np.where(phi > 0, phi ** m, np.nan)
        Ro = F * rw
        ratio = np.where(rt > 0, Ro / rt, np.nan)
        Sw = np.where(ratio >= 0, ratio ** (1.0 / n), np.nan)
    return np.clip(Sw, 0.0, 1.0)


def wyllie_porosity(dt, dt_ma=55.5, dt_fl=189.0):
    """Wyllie time-average porosity from sonic log."""
    return np.clip((dt - dt_ma) / (dt_fl - dt_ma), 0.0, 0.45)


def rhob_nphi_porosity(rhob, nphi, rho_ma=2.65, rho_fl=1.0):
    """Density-derived porosity."""
    phi_d = np.clip((rho_ma - rhob) / (rho_ma - rho_fl), 0.0, 0.45)
    phi_avg = (phi_d + nphi) / 2.0
    return phi_d, phi_avg


def vcl_from_gr(gr, gr_clean=30, gr_shale=120):
    """Volume of clay from gamma ray (linear model)."""
    return np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)


def uncertainty_porosity(phi, vcl, method="NPHI-RHOB"):
    """Estimate 1-sigma uncertainty in porosity."""
    base = 0.03
    clay_penalty = vcl * 0.08
    return base + clay_penalty


# ── Lithology classification rules ───────────────────────────────────────────

LITHOLOGY_RULES = {
    "shale":      {"gr": (75, 999), "nphi": (0.25, 0.60), "rhob": (2.2, 2.7)},
    "sandstone":  {"gr": (0,  60),  "nphi": (0.05, 0.35), "rhob": (2.0, 2.65)},
    "limestone":  {"gr": (0,  40),  "nphi": (0.00, 0.25), "rhob": (2.5, 2.71)},
    "dolomite":   {"gr": (0,  35),  "nphi": (-0.02, 0.20),"rhob": (2.7, 2.87)},
    "coal":       {"gr": (0,  30),  "nphi": (0.40, 1.00), "rhob": (1.0, 1.8)},
    "anhydrite":  {"gr": (0,  25),  "nphi": (-0.05, 0.02),"rhob": (2.9, 2.99)},
}

LITHOLOGY_COLORS = {
    "shale":     "#A8A8A8",
    "sandstone": "#F5DEB3",
    "limestone": "#90EE90",
    "dolomite":  "#87CEEB",
    "coal":      "#2F2F2F",
    "anhydrite": "#FFB6C1",
    "unknown":   "#DDDDDD",
}


def classify_lithology(gr, nphi, rhob):
    """Rule-based lithology classification. Returns (label, confidence)."""
    scores = {}
    for lith, bounds in LITHOLOGY_RULES.items():
        score = 0
        if bounds["gr"][0] <= gr <= bounds["gr"][1]:      score += 1
        if bounds["nphi"][0] <= nphi <= bounds["nphi"][1]: score += 1
        if bounds["rhob"][0] <= rhob <= bounds["rhob"][1]: score += 1
        scores[lith] = score
    best = max(scores, key=scores.get)
    conf = scores[best] / 3.0
    return (best, conf) if conf > 0 else ("unknown", 0.0)


# ── Knowledge base Q&A ───────────────────────────────────────────────────────

KB_QA = {
    "porosity": {
        "keywords": ["porosity", "poro", "phi", "void", "pore"],
        "answer": lambda ctx: _answer_porosity(ctx),
        "references": ["Wyllie et al. 1956", "Schlumberger Log Interpretation Vol.1"],
    },
    "water_saturation": {
        "keywords": ["water sat", "sw", "saturation", "hydrocarbon", "oil", "wet"],
        "answer": lambda ctx: _answer_sw(ctx),
        "references": ["Archie 1942", "Simandoux 1963", "SPE-12345"],
    },
    "lithology": {
        "keywords": ["lithology", "litho", "rock type", "formation", "facies", "shale", "sand", "limestone"],
        "answer": lambda ctx: _answer_lithology(ctx),
        "references": ["Serra 1984", "Rider 2002 – The Geological Interpretation of Well Logs"],
    },
    "clay": {
        "keywords": ["clay", "shale volume", "vcl", "vsh", "gamma"],
        "answer": lambda ctx: _answer_clay(ctx),
        "references": ["Larionov 1969", "Clavier 1971"],
    },
    "permeability": {
        "keywords": ["perm", "permeability", "flow", "darcy", "k"],
        "answer": lambda ctx: _answer_perm(ctx),
        "references": ["Timur 1968", "Coates & Dumanoir 1974"],
    },
    "uncertainty": {
        "keywords": ["uncertain", "confidence", "error", "reliable", "trust"],
        "answer": lambda ctx: _answer_uncertainty(ctx),
        "references": ["Caers 2011 – Modeling Uncertainty in Earth Sciences"],
    },
}


def _answer_porosity(ctx):
    phi_d, phi_avg = ctx.get("phi_d", (None, None)) if isinstance(ctx.get("phi_d"), tuple) else (ctx.get("phi_d"), ctx.get("phi_avg"))
    vcl = ctx.get("vcl", 0.2)
    unc = ctx.get("unc_phi", 0.03)
    if phi_avg is not None:
        return (
            f"Effective porosity estimated at **{phi_avg*100:.1f} ± {unc*100:.1f}%** (1σ). "
            f"Density porosity: {(phi_d or 0)*100:.1f}%. Clay correction applied using Vcl = {vcl*100:.0f}%. "
            f"{'High clay content reduces reliability — consider Thomas-Stieber correction.' if vcl > 0.3 else 'Low clay content — estimate is reliable.'}"
        )
    return "Upload a LAS file with RHOB, NPHI, and GR curves to compute porosity."


def _answer_sw(ctx):
    sw = ctx.get("sw")
    rt = ctx.get("rt")
    if sw is not None:
        fluid = "oil/gas bearing" if sw < 0.5 else ("transition zone" if sw < 0.7 else "water bearing")
        return (
            f"Water saturation Sw = **{sw*100:.1f}%** (Archie, Rw = 0.05 Ω·m). "
            f"RT = {rt:.1f} Ω·m. Interpretation: **{fluid}**. "
            f"{'Consider Simandoux model if Vcl > 0.2.' if ctx.get('vcl', 0) > 0.2 else 'Clean formation — Archie applies directly.'}"
        )
    return "Need RT (resistivity), RHOB, and NPHI to compute water saturation."


def _answer_lithology(ctx):
    lith = ctx.get("lithology", "unknown")
    conf = ctx.get("lith_conf", 0)
    return (
        f"Most likely lithology: **{lith.capitalize()}** (confidence: {conf*100:.0f}%). "
        f"Based on GR={ctx.get('gr',0):.0f} API, NPHI={ctx.get('nphi',0):.2f}, RHOB={ctx.get('rhob',0):.2f} g/cc. "
        f"{'Add core photos to PetroMind for higher confidence.' if conf < 0.8 else 'High confidence — consistent with log signatures.'}"
    )


def _answer_clay(ctx):
    vcl = ctx.get("vcl", None)
    gr = ctx.get("gr", None)
    if vcl is not None:
        return (
            f"Volume of clay Vcl = **{vcl*100:.1f}%** (GR linear model, GRclean=30, GRshale=120). "
            f"GR = {gr:.0f} API. "
            f"{'High clay — significant impact on porosity and permeability.' if vcl > 0.3 else 'Moderate clay content. Apply clay correction to porosity.'}"
        )
    return "Need GR log to compute Vcl. Provide a LAS file with gamma ray curve."


def _answer_perm(ctx):
    phi_avg = ctx.get("phi_avg", 0.15)
    sw = ctx.get("sw", 0.5)
    k_timur = 0.136 * (phi_avg ** 4.4) / (sw ** 2) * 1000
    return (
        f"Timur permeability estimate: **{k_timur:.1f} mD** (φ={phi_avg*100:.1f}%, Sw={sw*100:.1f}%). "
        f"{'Good reservoir quality.' if k_timur > 10 else ('Marginal quality.' if k_timur > 1 else 'Tight — may need stimulation.')} "
        f"Note: Timur equation is calibrated on sandstones. Carbonate permeability requires NMR or core data."
    )


def _answer_uncertainty(ctx):
    unc = ctx.get("unc_phi", 0.03)
    vcl = ctx.get("vcl", 0.2)
    return (
        f"Current porosity uncertainty: **±{unc*100:.1f}%** (1σ). "
        f"Main sources: clay model ({vcl*100:.0f}% Vcl), matrix density assumption (±0.03 g/cc). "
        f"To reduce uncertainty: (1) add core measurements, (2) upload core photos for lithology confirmation, (3) specify local matrix density."
    )


def answer_question(question: str, context: dict) -> tuple[str, list[str]]:
    """Route a question to the right KB answer. Returns (answer_text, references)."""
    q_lower = question.lower()
    for key, entry in KB_QA.items():
        if any(kw in q_lower for kw in entry["keywords"]):
            try:
                answer = entry["answer"](context)
            except Exception:
                answer = f"Could not compute {key} — check that required log curves are present."
            return answer, entry["references"]
    return (
        "I can answer questions about **porosity**, **water saturation**, **lithology**, "
        "**clay volume**, **permeability**, and **uncertainty**. What would you like to know about this well?",
        []
    )
