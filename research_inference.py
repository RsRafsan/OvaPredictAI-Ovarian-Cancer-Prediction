"""OvaPredict AI: exact v4 research-bundle inference. No clinical probability mapping.

Input: 49 raw predictors, in any order, one row per record.
The saved fitted preprocessor and trained model are loaded from one bundle.
This module does not refit any transformation or send data to a server.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import pickle
import numpy as np
import pandas as pd
# Needed by older sklearn installations to enable deserialization of IterativeImputer.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

MODEL_PATH = Path(__file__).resolve().parent / "models" / "simulated_federated_v4_research_bundle.pkl"
EXPECTED_SHA256 = "34bf684df6fba30c2523887d0f57a7de6227131444496725dc62c00a1a2713d4"


@lru_cache(maxsize=1)
def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle missing: {path}")
    # Never load model pickles uploaded by website visitors.
    if path.resolve() == MODEL_PATH.resolve():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError("Model checksum differs from the research-verified bundle.")
    with path.open("rb") as fh:
        bundle = pickle.load(fh)  # Load only the packaged, trusted model file.
    expected = {'preprocessing', 'rff_mapper', 'weights', 'bias',
                'training_score_mean', 'training_score_std', 'standardized_threshold'}
    if not expected.issubset(bundle):
        raise ValueError("The model bundle is incomplete.")
    return bundle


def raw_feature_names(bundle: dict) -> list[str]:
    return list(bundle['preprocessing']['pre_state']['feature_order'])


def selected_feature_names(bundle: dict) -> list[str]:
    return list(bundle['preprocessing']['features'])


def example_defaults(bundle: dict) -> dict[str, float]:
    """Only for DEMONSTRATION, from fitted development-set median imputer."""
    st = bundle['preprocessing']['pre_state']
    stats = st['simple_imputer'].statistics_
    defaults = dict(zip(st['regular_cols'], [float(x) for x in stats]))
    # CA72-4 missingness is handled by the saved iterative imputer.
    defaults['CA72-4'] = float('nan')
    return defaults


def validate_raw_data(raw: pd.DataFrame, bundle: dict, *,
                      allow_ca72_omitted: bool = True) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Provide a non-empty table of raw measurement values.")
    cols = raw_feature_names(bundle)
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not supported.")
    missing = [c for c in cols if c not in raw.columns]
    if allow_ca72_omitted:
        missing = [c for c in missing if c != 'CA72-4']
    if missing:
        raise ValueError("Missing required raw input columns: " + ", ".join(missing) + ". Download the 49-feature template.")
    df = raw.reindex(columns=cols).copy()
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric value in {c!r}; use numbers or empty cells.") from exc
    if not np.isfinite(df.to_numpy(dtype=float)[~df.isna().to_numpy()]).all():
        raise ValueError("Input contains infinite measurements.")
    valid_menopause = df['Menopause'].dropna().isin([0, 1]).all()
    if not valid_menopause:
        raise ValueError("Menopause must be 0 (No), 1 (Yes), or empty.")
    return df


def research_federated_predict(raw: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Return binary model labels and standardized SVM margins, NOT probabilities."""
    if bundle is None:
        bundle = load_bundle()
    df = validate_raw_data(raw, bundle)
    pre = bundle['preprocessing']; st = pre['pre_state']
    cols = list(st['feature_order'])
    reg = pd.DataFrame(st['simple_imputer'].transform(df[st['regular_cols']]),
                       columns=st['regular_cols'], index=df.index)
    if 'CA72-4' in cols:
        z = reg.copy()
        z['CA72-4'] = df['CA72-4'].to_numpy()
        pp = pd.DataFrame(st['iterative_imputer'].transform(z),
                          columns=z.columns, index=df.index)[cols]
    else:
        pp = reg[cols]
    for name, (lower, upper) in st['clipping_limits'].items():
        pp[name] = pp[name].clip(lower, upper)
    pp = pp[pre['features']].copy()
    if pre['continuous_cols']:
        pp[pre['continuous_cols']] = pre['scaler'].transform(pp[pre['continuous_cols']])
    z_rff = bundle['rff_mapper'].transform(np.asarray(pp, dtype=np.float64))
    margins = z_rff @ bundle['weights'] + bundle['bias']
    score_mean = float(bundle['training_score_mean'])
    score_std = max(float(bundle['training_score_std']), 1e-8)
    score_z = (margins - score_mean) / score_std
    threshold = float(bundle['standardized_threshold'])
    binary = (score_z >= threshold).astype(int)
    return pd.DataFrame({
        'Model Class': np.where(binary == 1, 'Malignant', 'Benign'),
        'Class Code': binary,
        'Standardized Decision Score': score_z,
        'Decision Threshold': np.full(len(score_z), threshold),
    }, index=df.index)


def make_report_pdf(raw: pd.DataFrame, prediction: pd.DataFrame) -> bytes:
    """Research-only PDF summary. Numeric margin is not a calibrated probability."""
    from fpdf import FPDF
    row = prediction.iloc[0]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'OvaPredict AI', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, 'Ovarian Cancer Classification - Research Prototype', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)
    pdf.set_fill_color(235, 242, 249)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 9, f"Model classification: {row['Model Class']}", fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, f"Standardized decision score: {float(row['Standardized Decision Score']):+.4f}", new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 7, f"Saved decision threshold: {float(row['Decision Threshold']):+.4f}", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.multi_cell(0, 5, 'Research demonstration only. The score is an SVM margin, not a clinical risk percentage or validated probability. Not for diagnosis, screening, or treatment decisions.')
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Input measurements', new_x='LMARGIN', new_y='NEXT')
    for key, val in raw.iloc[0].items():
        pdf.set_font('Helvetica', '', 9)
        value = 'Not measured (imputed)' if pd.isna(val) else f'{float(val):.4g}'
        pdf.cell(77, 6, str(key), border=1)
        pdf.cell(90, 6, value, border=1, new_x='LMARGIN', new_y='NEXT')
    return bytes(pdf.output())
