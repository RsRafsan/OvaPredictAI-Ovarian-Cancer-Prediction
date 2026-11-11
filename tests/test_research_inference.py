"""Regression checks based on frozen outputs of executed Kaggle research inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_inference import (load_bundle, research_federated_predict,
    raw_feature_names, validate_raw_data, make_report_pdf)

FIX=Path(__file__).with_name('golden_research_inference.json')

@pytest.fixture(scope='module')
def model():
    b = load_bundle()
    # For local verification only: fitted sklearn 1.6.1 artifacts may need a
    # dtype compatibility attribute under sklearn 1.8+ (sandbox installation).
    # Production requirements pin the actual training version (1.6.1).
    for imp in (b['preprocessing']['pre_state']['simple_imputer'],
                b['preprocessing']['pre_state']['iterative_imputer'].initial_imputer_):
        if imp is not None and not hasattr(imp, '_fill_dtype'):
            imp._fill_dtype = np.float64
    return b

@pytest.fixture(scope='module')
def sample():
    return json.loads(FIX.read_text())['cases']


def test_trained_artifact_present_and_features(model):
    assert len(raw_feature_names(model))==49
    assert len(model['preprocessing']['features'])==25
    assert abs(float(model['standardized_threshold'])-0.026657149543612844)<1e-12


def test_inference_golden_examples(model,sample):
    df=pd.DataFrame([x['values'] for x in sample])
    got=research_federated_predict(df,model)
    np.testing.assert_allclose(got['Standardized Decision Score'],
        [x['expected_score'] for x in sample],atol=1e-10,rtol=0)
    assert got['Class Code'].tolist()==[x['expected_class'] for x in sample]
    assert set(got['Model Class'])<= {'Benign','Malignant'}
    assert 'risk' not in ' '.join(got.columns).lower()
    assert 'probability' not in ' '.join(got.columns).lower()


def test_input_column_order_permutation(model,sample):
    df=pd.DataFrame([x['values'] for x in sample])
    a=research_federated_predict(df,model)
    b=research_federated_predict(df[list(reversed(df.columns))],model)
    np.testing.assert_allclose(a['Standardized Decision Score'],b['Standardized Decision Score'],atol=1e-12)


def test_ca72_missing_is_supported(model,sample):
    df=pd.DataFrame([x['values'] for x in sample])
    df=df.drop(columns=['CA72-4'])
    result=research_federated_predict(df,model)
    assert len(result)==len(sample)
    assert np.isfinite(result['Standardized Decision Score']).all()


def test_missing_other_input_col_rejected(model,sample):
    df=pd.DataFrame([x['values'] for x in sample]).drop(columns=['HE4'])
    with pytest.raises(ValueError, match='HE4'):
        research_federated_predict(df,model)


def test_bad_inputs_rejected(model,sample):
    df=pd.DataFrame([x['values'] for x in sample])
    bad=df.astype({'HE4':object}).copy();bad.loc[0,'HE4']='unrecorded'
    with pytest.raises(ValueError, match='HE4'):
        research_federated_predict(bad,model)
    bad=df.copy();bad.loc[0,'Menopause']=5
    with pytest.raises(ValueError, match='Menopause'):
        research_federated_predict(bad,model)
    bad=df.copy();bad.loc[0,'HE4']=float('inf')
    with pytest.raises(ValueError, match='infinite'):
        research_federated_predict(bad,model)


def test_csv_batch_export(model,sample,tmp_path):
    df=pd.DataFrame([x['values'] for x in sample]); source=tmp_path/'input.csv'
    df.to_csv(source,index=False)
    reread=pd.read_csv(source)
    out=research_federated_predict(reread,model)
    result=reread.copy()
    for c in out.columns:result[c]=out[c].to_numpy()
    target=tmp_path/'predictions.csv'; result.to_csv(target,index=False)
    assert len(pd.read_csv(target))==len(sample)
    assert 'Standardized Decision Score' in pd.read_csv(target).columns


def test_pdf_report(model,sample):
    raw=pd.DataFrame([sample[0]['values']])
    pred=research_federated_predict(raw,model)
    pdf=make_report_pdf(raw,pred)
    assert pdf[:5]==b'%PDF-'
    assert len(pdf)>1500
