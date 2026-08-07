"""
GlobalWellFM — HuggingFace model connector.
Downloads model from Aljaser2030/GlobalWellFM and runs lithofacies prediction.
Falls back to physics-only if model unavailable.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

_MODEL = None
_IMPUTER = None
_LE = None
_LOADED = False

FEATURE_COLS = [
    'GR_N', 'RHOB_N', 'NPHI_N', 'RT_LOG',
    'VCL', 'PHI_D', 'PHI_N', 'PHI_AVG', 'PHI_EFF',
    'NPHI_RHOB_DIFF', 'GR_RT_RATIO', 'SW'
]

HF_REPO = 'Aljaser2030/GlobalWellFM'


def _load_model():
    global _MODEL, _IMPUTER, _LE, _LOADED
    if _LOADED:
        return _MODEL is not None
    _LOADED = True
    try:
        from huggingface_hub import hf_hub_download
        import joblib
        import socket
        # 10-second timeout — if HuggingFace is slow, use physics fallback immediately
        socket.setdefaulttimeout(10)
        model_path   = hf_hub_download(HF_REPO, 'model/xgb_model.joblib')
        imputer_path = hf_hub_download(HF_REPO, 'model/imputer.joblib')
        le_path      = hf_hub_download(HF_REPO, 'model/label_encoder.joblib')
        socket.setdefaulttimeout(None)
        _MODEL   = joblib.load(model_path)
        _IMPUTER = joblib.load(imputer_path)
        _LE      = joblib.load(le_path)
        return True
    except Exception as e:
        print(f'[GlobalWellFM] Model load failed: {e}  — using physics fallback')
        socket.setdefaulttimeout(None)
        return False


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f['GR_N']   = f['GR'].clip(0, 200) / 200 if 'GR' in f.columns else 0.5
    f['RHOB_N'] = (f['RHOB'].clip(1.5, 3.0) - 1.5) / 1.5 if 'RHOB' in f.columns else 0.5
    f['NPHI_N'] = f['NPHI'].clip(-0.1, 0.6) if 'NPHI' in f.columns else 0.25
    rt = f['RT'].clip(0.01, 10000) if 'RT' in f.columns else pd.Series(np.ones(len(f)) * 10)
    f['RT_LOG'] = np.log10(rt)
    f['VCL']    = ((f.get('GR', 75).clip(0, 200) - 30) / (120 - 30)).clip(0, 1)
    f['PHI_D']  = ((2.65 - f.get('RHOB', 2.4)) / (2.65 - 1.0)).clip(0, 0.45)
    f['PHI_N']  = f.get('NPHI', 0.2).clip(0, 0.45)
    f['PHI_AVG']= (f['PHI_D'] + f['PHI_N']) / 2
    f['PHI_EFF']= (f['PHI_AVG'] * (1 - f['VCL'])).clip(0, 0.45)
    f['NPHI_RHOB_DIFF'] = f['NPHI_N'] - f['RHOB_N']
    f['GR_RT_RATIO']    = f['GR_N'] / (f['RT_LOG'] + 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Ro = 0.05 / np.power(np.where(f['PHI_EFF'] > 0, f['PHI_EFF'], np.nan), 2)
        f['SW'] = np.clip(np.sqrt(Ro / rt.values), 0, 1)
    return f


def predict(df: pd.DataFrame):
    """
    Run GlobalWellFM lithofacies prediction on an interpreted DataFrame.
    Returns (lithology_series, confidence_series, model_used_str)
    """
    available = _load_model()

    if available and _MODEL is not None:
        try:
            feat = _engineer(df)
            X = feat[FEATURE_COLS].values
            X = _IMPUTER.transform(X)
            preds  = _MODEL.predict(X)
            probs  = _MODEL.predict_proba(X)
            labels = _LE.inverse_transform(preds)
            conf   = probs.max(axis=1)
            return pd.Series(labels, index=df.index), pd.Series(conf, index=df.index), 'GlobalWellFM'
        except Exception as e:
            print(f'[GlobalWellFM] Prediction failed: {e}')

    # Physics fallback
    feat = _engineer(df)
    vcl  = feat['VCL']
    phi  = feat['PHI_EFF']
    sw   = feat.get('SW', pd.Series(np.ones(len(df)) * 0.8))
    rhob = df.get('RHOB', pd.Series(np.ones(len(df)) * 2.45))

    def classify(row):
        v, p, s, r = row['VCL'], row['PHI_EFF'], row['SW'], row['RHOB']
        if r < 1.8:  return 'Coal'
        if v > 0.65: return 'Shale'
        if r > 2.80: return 'Anhydrite' if p < 0.04 else 'Dolomite'
        if r > 2.63: return 'Limestone'
        return 'Sandstone'

    combined = pd.DataFrame({'VCL': vcl, 'PHI_EFF': phi, 'SW': sw, 'RHOB': rhob})
    lith = combined.apply(classify, axis=1)
    conf = pd.Series(np.ones(len(df)) * 0.72, index=df.index)
    return lith, conf, 'Physics Rules'


def model_status():
    """Returns dict with model info for display."""
    ok = _load_model()
    if ok and _MODEL is not None:
        return {
            'loaded': True,
            'name': 'GlobalWellFM',
            'repo': HF_REPO,
            'f1': 0.9494,
            'classes': list(_LE.classes_) if _LE else [],
            'source': f'🤗 huggingface.co/{HF_REPO}',
        }
    return {
        'loaded': False,
        'name': 'Physics Rules (fallback)',
        'repo': None,
        'f1': None,
        'classes': ['Shale', 'Sandstone', 'Limestone', 'Dolomite', 'Coal', 'Anhydrite'],
        'source': 'Built-in petrophysics',
    }
