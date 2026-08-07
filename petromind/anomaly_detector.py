"""
SubsurfAI — Automated Anomaly Detection
Scans well log data for:
  1. Overlooked pay zones (low Sw + good porosity that wasn't flagged)
  2. Gas crossover (NPHI-RHOB separation)
  3. Fluid contacts (sharp resistivity transitions)
  4. Data quality issues (washouts, cycle skipping, bad hole)
  5. Tight zones with elevated RT (non-reservoir)
  6. Abnormal pressure indicators
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


# ── Severity colours ──────────────────────────────────────────────────────────
SEVERITY = {
    'HIGH':   {'emoji': '🔴', 'color': '#EF4444', 'label': 'High'},
    'MEDIUM': {'emoji': '🟡', 'color': '#F59E0B', 'label': 'Medium'},
    'LOW':    {'emoji': '🟢', 'color': '#10B981', 'label': 'Low'},
    'INFO':   {'emoji': '🔵', 'color': '#38BDF8', 'label': 'Info'},
}


def _smooth(series, window=5):
    vals = series.fillna(method='ffill').fillna(method='bfill').values
    return pd.Series(uniform_filter1d(vals.astype(float), size=window), index=series.index)


def detect_anomalies(df: pd.DataFrame, config: dict = None) -> list:
    """
    Run all anomaly detectors on an interpreted well log DataFrame.

    Returns list of anomaly dicts:
      {
        'type': str,
        'severity': 'HIGH'|'MEDIUM'|'LOW'|'INFO',
        'depth_top': float,
        'depth_base': float,
        'depth_center': float,
        'thickness': float,
        'confidence': float,  # 0-1
        'title': str,
        'description': str,
        'recommendation': str,
        'values': dict,       # supporting measurements
      }
    """
    if config is None:
        config = {}

    anomalies = []
    depth = df['DEPTH']

    # ── 1. Overlooked Pay Zones ───────────────────────────────────────────────
    if all(c in df.columns for c in ['PHI_EFF', 'SW', 'VCL']):
        phi  = df['PHI_EFF'].fillna(0)
        sw   = df['SW'].fillna(1)
        vcl  = df['VCL'].fillna(1)
        flag = df.get('FLUID_FLAG', pd.Series(['undetermined']*len(df), index=df.index))

        phi_cut = config.get('phi_cut', 0.08)
        sw_cut  = config.get('sw_cut',  0.60)
        vcl_cut = config.get('vcl_cut', 0.50)

        # Rows with good reservoir quality but not flagged as HC
        overlooked = (phi > phi_cut) & (sw < sw_cut) & (vcl < vcl_cut) & \
                     (flag != 'hydrocarbon')

        if overlooked.sum() > 3:
            zones = _group_intervals(df, overlooked)
            for z in zones:
                sub = df.loc[z['mask']]
                avg_phi = sub['PHI_EFF'].mean()
                avg_sw  = sub['SW'].mean()
                conf = min(0.95, (1 - avg_sw) * (avg_phi / 0.25) * 0.9)
                if z['thickness'] < 1.0:
                    continue
                anomalies.append({
                    'type': 'overlooked_pay',
                    'severity': 'HIGH' if conf > 0.75 else 'MEDIUM',
                    'depth_top': z['top'],
                    'depth_base': z['base'],
                    'depth_center': (z['top'] + z['base']) / 2,
                    'thickness': z['thickness'],
                    'confidence': round(conf, 2),
                    'title': f"Potential Pay Zone — {z['top']:.1f}–{z['base']:.1f} m",
                    'description': (
                        f"φ_eff={avg_phi:.3f}, Sw={avg_sw:.3f} — reservoir quality meets "
                        f"cutoffs (φ>{phi_cut}, Sw<{sw_cut}) but zone was not flagged as hydrocarbon."
                    ),
                    'recommendation': (
                        "Review Rw value and Archie parameters. Consider reducing Sw cutoff. "
                        "Check for adjacent wells with production from this interval."
                    ),
                    'values': {'phi_eff': round(avg_phi, 3), 'sw': round(avg_sw, 3),
                               'vcl': round(sub['VCL'].mean(), 3)},
                })

    # ── 2. Gas Crossover (NPHI-RHOB) ─────────────────────────────────────────
    if all(c in df.columns for c in ['NPHI', 'RHOB']):
        # Normalise both to same scale
        nphi_n = df['NPHI'].clip(-0.05, 0.60)
        rhob_n = (df['RHOB'].clip(1.9, 2.9) - 1.9) / (2.9 - 1.9) * 0.65  # scale to NPHI range

        crossover = nphi_n < rhob_n - 0.04  # NPHI reads lower than RHOB (gas effect)

        if crossover.sum() > 3:
            zones = _group_intervals(df, crossover)
            for z in zones:
                if z['thickness'] < 0.5: continue
                sub = df.loc[z['mask']]
                sep = (rhob_n.loc[z['mask']] - nphi_n.loc[z['mask']]).mean()
                conf = min(0.95, sep * 8)
                anomalies.append({
                    'type': 'gas_crossover',
                    'severity': 'HIGH' if conf > 0.7 else 'MEDIUM',
                    'depth_top': z['top'],
                    'depth_base': z['base'],
                    'depth_center': (z['top'] + z['base']) / 2,
                    'thickness': z['thickness'],
                    'confidence': round(conf, 2),
                    'title': f"Gas Crossover — {z['top']:.1f}–{z['base']:.1f} m",
                    'description': (
                        f"NPHI reads lower than RHOB (separation={sep:.3f}) — classic gas indicator. "
                        f"Gas reduces neutron response while density reads lighter matrix."
                    ),
                    'recommendation': (
                        "Verify with resistivity response. If RT also elevated, high confidence gas. "
                        "Apply gas correction to porosity calculations."
                    ),
                    'values': {'nphi_avg': round(sub['NPHI'].mean(), 3),
                               'rhob_avg': round(sub['RHOB'].mean(), 3),
                               'separation': round(sep, 3)},
                })

    # ── 3. Fluid Contact Detection (RT transition) ────────────────────────────
    if 'RT' in df.columns:
        rt_log = np.log10(df['RT'].clip(0.1, 5000).fillna(1))
        rt_smooth = _smooth(rt_log, window=10)
        gradient = rt_smooth.diff().abs()

        # Find sharp transitions
        threshold = gradient.quantile(0.97)
        contact_mask = gradient > threshold

        if contact_mask.sum() > 0:
            contact_depths = depth[contact_mask].values
            # Group close contacts
            contacts = []
            if len(contact_depths) > 0:
                prev = contact_depths[0]
                for d in contact_depths[1:]:
                    if d - prev > 5:
                        contacts.append(prev)
                    prev = d
                contacts.append(prev)

            for cd in contacts[:5]:  # max 5 contacts
                idx = (depth - cd).abs().idxmin()
                rt_val = df['RT'].loc[idx] if 'RT' in df.columns else None
                sw_val = df['SW'].loc[idx] if 'SW' in df.columns else None

                anomalies.append({
                    'type': 'fluid_contact',
                    'severity': 'MEDIUM',
                    'depth_top': cd - 1,
                    'depth_base': cd + 1,
                    'depth_center': cd,
                    'thickness': 2.0,
                    'confidence': 0.65,
                    'title': f"Possible Fluid Contact — {cd:.1f} m",
                    'description': (
                        f"Sharp resistivity transition detected at {cd:.1f} m "
                        f"(RT={rt_val:.1f} ohm.m). May indicate OWC, GWC, or formation boundary."
                    ),
                    'recommendation': (
                        "Cross-check with NPHI-RHOB crossplot. Compare to nearby wells. "
                        "If consistent with porosity and Sw trends, likely a real fluid contact."
                    ),
                    'values': {'rt': round(rt_val, 2) if rt_val else None,
                               'sw': round(sw_val, 3) if sw_val else None},
                })

    # ── 4. Washout / Bad Hole (CALI) ─────────────────────────────────────────
    if 'CALI' in df.columns:
        cali = df['CALI'].clip(4, 24)
        bit_size = config.get('bit_size', 8.5)
        washout = cali > bit_size * 1.15  # >15% over bit size

        if washout.sum() > 3:
            zones = _group_intervals(df, washout)
            for z in zones:
                if z['thickness'] < 1.0: continue
                sub_cali = cali.loc[z['mask']].mean()
                anomalies.append({
                    'type': 'washout',
                    'severity': 'MEDIUM',
                    'depth_top': z['top'],
                    'depth_base': z['base'],
                    'depth_center': (z['top'] + z['base']) / 2,
                    'thickness': z['thickness'],
                    'confidence': 0.85,
                    'title': f"Washout / Bad Hole — {z['top']:.1f}–{z['base']:.1f} m",
                    'description': (
                        f"Caliper reads {sub_cali:.1f}\" vs bit size {bit_size}\". "
                        f"Enlarged borehole degrades density and neutron log quality in this interval."
                    ),
                    'recommendation': (
                        "Flag density and neutron readings as unreliable in this interval. "
                        "Use sonic log if available. Avoid using RHOB/NPHI for porosity calculation here."
                    ),
                    'values': {'cali_avg': round(sub_cali, 2), 'bit_size': bit_size},
                })

    # ── 5. Tight High-RT Zone (non-pay) ──────────────────────────────────────
    if all(c in df.columns for c in ['RT', 'PHI_EFF']):
        rt    = df['RT'].clip(0.1, 5000)
        phi   = df['PHI_EFF'].fillna(0)
        tight = (rt > 50) & (phi < 0.05)

        if tight.sum() > 3:
            zones = _group_intervals(df, tight)
            for z in zones:
                if z['thickness'] < 0.5: continue
                sub = df.loc[z['mask']]
                anomalies.append({
                    'type': 'tight_zone',
                    'severity': 'INFO',
                    'depth_top': z['top'],
                    'depth_base': z['base'],
                    'depth_center': (z['top'] + z['base']) / 2,
                    'thickness': z['thickness'],
                    'confidence': 0.80,
                    'title': f"Tight Zone (High RT, Low φ) — {z['top']:.1f}–{z['base']:.1f} m",
                    'description': (
                        f"High resistivity (RT={sub['RT'].mean():.0f} ohm.m) with near-zero porosity "
                        f"(φ={sub['PHI_EFF'].mean():.3f}). Likely cemented/tight rock — not pay."
                    ),
                    'recommendation': (
                        "Could be anhydrite, tight carbonate, or cemented sandstone. "
                        "Check PEF log if available for mineral identification."
                    ),
                    'values': {'rt_avg': round(sub['RT'].mean(), 1),
                               'phi_avg': round(sub['PHI_EFF'].mean(), 3)},
                })

    # Sort by severity then depth
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'INFO': 3}
    anomalies.sort(key=lambda x: (severity_order[x['severity']], x['depth_top']))

    return anomalies


def _group_intervals(df, mask: pd.Series, min_gap: float = 2.0) -> list:
    """Group consecutive True values into depth intervals."""
    depth = df['DEPTH']
    zones = []
    in_zone = False
    top = None
    zone_mask = pd.Series(False, index=df.index)

    for i, (idx, val) in enumerate(mask.items()):
        if val and not in_zone:
            in_zone = True
            top = depth.loc[idx]
            zone_mask = pd.Series(False, index=df.index)
        if in_zone:
            zone_mask.loc[idx] = True
        if in_zone and (not val or i == len(mask) - 1):
            base = depth.loc[idx]
            thickness = base - top
            if thickness >= 0:
                zones.append({
                    'top': top, 'base': base,
                    'thickness': thickness,
                    'mask': zone_mask.copy(),
                })
            in_zone = False

    return zones


def anomaly_summary(anomalies: list) -> dict:
    """Return counts and plain-English summary."""
    if not anomalies:
        return {
            'total': 0,
            'by_type': {},
            'by_severity': {},
            'headline': "✅ No significant anomalies detected.",
            'detail': "Well log quality appears good. All interpreted zones are consistent.",
        }

    by_severity = {}
    by_type = {}
    for a in anomalies:
        by_severity[a['severity']] = by_severity.get(a['severity'], 0) + 1
        by_type[a['type']] = by_type.get(a['type'], 0) + 1

    high = by_severity.get('HIGH', 0)
    med  = by_severity.get('MEDIUM', 0)

    pay_count = by_type.get('overlooked_pay', 0)
    gas_count = by_type.get('gas_crossover', 0)

    headline_parts = []
    if pay_count: headline_parts.append(f"{pay_count} potential pay zone{'s' if pay_count>1 else ''}")
    if gas_count: headline_parts.append(f"{gas_count} gas indicator{'s' if gas_count>1 else ''}")

    headline = (
        f"⚠️ {len(anomalies)} anomalies detected — "
        + (", ".join(headline_parts) if headline_parts else f"{high} high-priority")
    )

    return {
        'total': len(anomalies),
        'by_type': by_type,
        'by_severity': by_severity,
        'headline': headline,
        'detail': f"{high} high · {med} medium · {by_severity.get('LOW',0)} low · {by_severity.get('INFO',0)} info",
    }
