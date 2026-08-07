"""
SubsurfAI — Missing Curve Prediction
Trained on 320,000 samples from 20 global wells.

Supported predictions:
  GR only          → predict RHOB, NPHI
  GR + RHOB        → predict NPHI, RT
  GR + NPHI        → predict RHOB, RT
  GR + RHOB + NPHI → predict RT
"""
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

_MODELS = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'curve_models', 'curve_prediction_models.joblib')

CURVE_RANGES = {
    'RHOB': (1.4, 3.2),
    'NPHI': (-0.05, 0.65),
    'RT':   (0.1, 5000),
    'DT':   (40, 250),
}

CURVE_UNITS = {
    'RHOB': 'g/c3',
    'NPHI': 'v/v',
    'RT':   'ohmm',
    'DT':   'us/f',
}


def _load():
    global _MODELS
    if _MODELS is None:
        if os.path.exists(_MODEL_PATH):
            _MODELS = joblib.load(_MODEL_PATH)
        else:
            _MODELS = {}
    return _MODELS


def detect_missing(df: pd.DataFrame) -> dict:
    """
    Analyse a DataFrame and return which important curves are missing or sparse.
    Returns dict: {curve: status} where status is 'missing', 'sparse', or 'ok'
    """
    result = {}
    important = ['GR', 'RHOB', 'NPHI', 'RT', 'DT', 'CALI']
    for curve in important:
        if curve not in df.columns:
            result[curve] = 'missing'
        else:
            pct = df[curve].notna().mean()
            if pct < 0.30:
                result[curve] = 'sparse'
            elif pct < 0.70:
                result[curve] = 'partial'
            else:
                result[curve] = 'ok'
    return result


def _predict_curve(model_key: str, df: pd.DataFrame, models: dict):
    """Run a single model prediction and return (predicted_series, uncertainty_series)."""
    if model_key not in models:
        return None, None
    m = models[model_key]
    inputs = m['inputs']

    # Check all inputs available
    missing_inputs = [c for c in inputs if c not in df.columns or df[c].isna().all()]
    if missing_inputs:
        return None, None

    feat_df = df[inputs].copy()
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN in inputs with column median
    for c in inputs:
        feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = m['scaler'].transform(feat_df.values)

    # Point prediction
    pred = m['model'].predict(X)

    # Uncertainty via tree variance (RandomForest)
    try:
        tree_preds = np.array([t.predict(X) for t in m['model'].estimators_])
        std = tree_preds.std(axis=0)
    except Exception:
        std = np.abs(pred) * 0.15

    if m.get('log', False):
        pred = 10 ** pred
        std  = pred * std * np.log(10)  # propagate uncertainty through exp

    # Clip to physical range
    curve_name = model_key.split('_from_')[0]
    lo, hi = CURVE_RANGES.get(curve_name, (-np.inf, np.inf))
    pred = np.clip(pred, lo, hi)
    std  = np.clip(std,  0, (hi - lo) * 0.3)

    return pd.Series(pred, index=df.index), pd.Series(std, index=df.index)


def predict_missing(df: pd.DataFrame) -> dict:
    """
    Predict all missing/sparse curves that can be estimated from available data.

    Returns dict:
      {
        'RHOB': {'predicted': Series, 'uncertainty': Series, 'model': str, 'r2': float},
        'NPHI': {...},
        'RT':   {...},
      }
    """
    models = _load()
    status = detect_missing(df)
    results = {}

    has = lambda c: c in df.columns and df[c].notna().mean() > 0.3

    # ── RHOB ──────────────────────────────────────────────────
    if status.get('RHOB') in ('missing', 'sparse'):
        if has('GR') and has('NPHI'):
            pred, unc = _predict_curve('RHOB_from_GR_NPHI', df, models)
            if pred is not None:
                results['RHOB'] = {'predicted': pred, 'uncertainty': unc,
                                   'model': 'GR+NPHI → RHOB',
                                   'r2': models['RHOB_from_GR_NPHI']['info']['r2'],
                                   'inputs': ['GR', 'NPHI']}
        elif has('GR'):
            pred, unc = _predict_curve('RHOB_from_GR', df, models)
            if pred is not None:
                results['RHOB'] = {'predicted': pred, 'uncertainty': unc,
                                   'model': 'GR → RHOB',
                                   'r2': models['RHOB_from_GR']['info']['r2'],
                                   'inputs': ['GR']}

    # ── NPHI ──────────────────────────────────────────────────
    if status.get('NPHI') in ('missing', 'sparse'):
        if has('GR') and has('RHOB'):
            pred, unc = _predict_curve('NPHI_from_GR_RHOB', df, models)
            if pred is not None:
                results['NPHI'] = {'predicted': pred, 'uncertainty': unc,
                                   'model': 'GR+RHOB → NPHI',
                                   'r2': models['NPHI_from_GR_RHOB']['info']['r2'],
                                   'inputs': ['GR', 'RHOB']}
        elif has('GR'):
            pred, unc = _predict_curve('NPHI_from_GR', df, models)
            if pred is not None:
                results['NPHI'] = {'predicted': pred, 'uncertainty': unc,
                                   'model': 'GR → NPHI',
                                   'r2': models['NPHI_from_GR']['info']['r2'],
                                   'inputs': ['GR']}

    # ── RT ────────────────────────────────────────────────────
    if status.get('RT') in ('missing', 'sparse'):
        if has('GR') and has('RHOB') and has('NPHI'):
            pred, unc = _predict_curve('RT_from_all', df, models)
            if pred is not None:
                results['RT'] = {'predicted': pred, 'uncertainty': unc,
                                 'model': 'GR+RHOB+NPHI → RT',
                                 'r2': models['RT_from_all']['info']['r2'],
                                 'inputs': ['GR', 'RHOB', 'NPHI']}

    return results


def apply_predictions(df: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """
    Merge predicted curves into the DataFrame.
    Predicted curves are stored as RHOB_PRED, NPHI_PRED, RT_PRED
    with uncertainty as RHOB_UNC, NPHI_UNC, RT_UNC.
    Original curves are preserved unchanged.
    """
    out = df.copy()
    for curve, data in predictions.items():
        out[f'{curve}_PRED'] = data['predicted']
        out[f'{curve}_UNC']  = data['uncertainty']
    return out


def prediction_summary(predictions: dict, status: dict) -> str:
    """Return a human-readable summary of what was predicted."""
    if not predictions:
        missing = [c for c, s in status.items() if s in ('missing', 'sparse')]
        if missing:
            return f"⚠️ Missing curves detected ({', '.join(missing)}) but insufficient input curves to predict them."
        return "✅ All key curves present — no prediction needed."

    lines = [f"🤖 **{len(predictions)} curve(s) predicted by AI:**"]
    for curve, data in predictions.items():
        lines.append(
            f"- **{curve}** from {data['inputs']} · R²={data['r2']} · model: _{data['model']}_"
        )
    lines.append("\n_Predicted curves shown with uncertainty ribbons. Export includes both real and predicted columns._")
    return "\n".join(lines)
