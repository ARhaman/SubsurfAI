"""
SubsurfAI — Claude-powered geoscience chat.
Sends interpreted well data as context to Claude claude-haiku-4-5-20251001 for expert answers.
"""
import os
import anthropic
import pandas as pd
import numpy as np


def _build_system_prompt() -> str:
    return """You are SubsurfAI, an expert petrophysicist and geoscientist with 20+ years of experience
in the Dutch North Sea (NLOG F-block, K-block, L-block). You have deep knowledge of:
- Petrophysical interpretation: Archie's law, Wyllie time-average, NPHI-RHOB crossplot
- Dutch North Sea geology: Rotliegend sandstones, Carboniferous, Chalk, Zechstein salt
- Reservoir characterization: porosity, permeability, water saturation, net pay
- Uncertainty quantification in well log interpretation
- Seismic facies classification

When answering:
- Be direct and specific — cite actual numbers from the well data provided
- Identify the best reservoir intervals by depth
- Flag hydrocarbon indicators (low Sw, elevated resistivity, crossover effects)
- Note data quality issues (washouts from CALI, cycle skipping in DT)
- Compare observations to typical Dutch North Sea expectations
- Give actionable recommendations
- Never say "I don't know" — reason from the data provided"""


def _summarize_well(df: pd.DataFrame, well_name: str, max_rows: int = 80) -> str:
    """Convert interpreted DataFrame to a compact text summary for Claude."""
    lines = [f"WELL: {well_name}"]
    lines.append(f"DEPTH RANGE: {df['DEPTH'].min():.1f} – {df['DEPTH'].max():.1f} m  ({len(df):,} samples)")

    # Curve stats
    curves = ["GR", "RHOB", "NPHI", "RT", "DT", "VCL", "PHI_EFF", "SW"]
    lines.append("\nCURVE STATISTICS (mean ± std | min | max):")
    for c in curves:
        if c in df.columns:
            v = df[c].dropna()
            if len(v):
                lines.append(f"  {c:8s}: {v.mean():.3f} ± {v.std():.3f}  |  {v.min():.3f} – {v.max():.3f}  ({len(v):,} valid)")

    # Lithology distribution
    if "LITHOLOGY" in df.columns:
        lines.append("\nLITHOLOGY DISTRIBUTION:")
        for lith, cnt in df["LITHOLOGY"].value_counts().items():
            pct = 100 * cnt / len(df)
            lines.append(f"  {lith:12s}: {cnt:5d} rows ({pct:.1f}%)")

    # Fluid flags
    if "FLUID_FLAG" in df.columns:
        lines.append("\nFLUID FLAG DISTRIBUTION:")
        for flag, cnt in df["FLUID_FLAG"].value_counts().items():
            pct = 100 * cnt / len(df)
            lines.append(f"  {flag:14s}: {cnt:5d} rows ({pct:.1f}%)")

    # Best reservoir intervals (top 5 by PHI_EFF where FLUID_FLAG == hydrocarbon)
    if "FLUID_FLAG" in df.columns and "PHI_EFF" in df.columns:
        hc = df[df["FLUID_FLAG"] == "hydrocarbon"].copy()
        if len(hc) > 0:
            lines.append(f"\nHYDROCARBON ZONES ({len(hc)} rows):")
            lines.append(f"  Depth: {hc['DEPTH'].min():.1f} – {hc['DEPTH'].max():.1f} m")
            lines.append(f"  Avg φ_eff: {hc['PHI_EFF'].mean():.3f}  |  Avg Sw: {hc['SW'].mean():.3f}")
            if "GR" in hc.columns:
                lines.append(f"  Avg GR: {hc['GR'].mean():.1f} API")
            if "RT" in hc.columns:
                lines.append(f"  Avg RT: {hc['RT'].mean():.2f} ohm.m")

    # Top reservoir rows (sorted by PHI_EFF)
    if "PHI_EFF" in df.columns:
        best = df.nlargest(min(max_rows, 30), "PHI_EFF")
        show_cols = [c for c in ["DEPTH","GR","RHOB","NPHI","RT","PHI_EFF","SW","LITHOLOGY","FLUID_FLAG"] if c in best.columns]
        lines.append(f"\nTOP {len(best)} ROWS BY POROSITY (φ_eff):")
        lines.append("  " + " | ".join(f"{c:>10}" for c in show_cols))
        for _, row in best.iterrows():
            vals = []
            for c in show_cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:>10.3f}")
                else:
                    vals.append(f"{str(v):>10}")
            lines.append("  " + " | ".join(vals))

    return "\n".join(lines)


def ask_claude(
    question: str,
    df: pd.DataFrame,
    well_name: str,
    chat_history: list[dict],
    api_key: str,
) -> str:
    """
    Send question + well context to Claude. Returns answer string.
    chat_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    client = anthropic.Anthropic(api_key=api_key)

    well_context = _summarize_well(df, well_name)

    # Build messages: inject well context into first user message
    messages = []
    for i, msg in enumerate(chat_history):
        if i == 0 and msg["role"] == "user":
            messages.append({
                "role": "user",
                "content": f"[WELL DATA CONTEXT]\n{well_context}\n\n[QUESTION]\n{msg['content']}"
            })
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    if not chat_history:
        messages.append({
            "role": "user",
            "content": f"[WELL DATA CONTEXT]\n{well_context}\n\n[QUESTION]\n{question}"
        })
    else:
        messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_build_system_prompt(),
        messages=messages,
    )

    return response.content[0].text
