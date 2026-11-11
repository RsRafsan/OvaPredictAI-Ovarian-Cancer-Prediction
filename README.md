# OvaPredict AI — Research Prototype

Streamlit ovarian-cancer classification research prototype aligned with the paper's saved v4-simulation federated RFF-hinge SVM.

## Install and start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Set Python 3.11 or 3.12 in the deployment environment. The model was saved with scikit-learn 1.6.1 (pinned above). The trained bundle is in `models/simulated_federated_v4_research_bundle.pkl`.

## Input and inference

- The fitted preprocessing requires 49 raw predictor variables. The classifier uses 25 selected features after imputation, clipping and scaling.
- The **Single Prediction** tab supplies all 49 inputs; fields start with *example training-set median* values for demonstration only. CA72-4 is missing by default and is estimated by the trained iterative imputer.
- The **Batch Prediction** tab accepts CSV files with the 49 raw column names (CA72-4 may be omitted or blank). Download the in-app CSV template.
- The score shown is a *standardized SVM margin*. `score >= saved_threshold` gives malignant (class 1); otherwise benign (class 0). It is **not** a calibrated probability or clinically validated risk percentage.
- No fitted preprocessing step is recomputed on visitor inputs. No records are written to persistent storage.
- The previously deployed 25-feature model, scaler, and SHAP CSV files are preserved in `legacy_reference/` for traceability, but are deliberately **not** used by current predictions. The manuscript's SHAP/LIME analyses describe the original centralized SVM, not this federated model.

## Tests

```bash
python -m pip install pytest
python -m pytest -q tests
```

Tests check saved-bundle checksum, raw input validation, patient-level prediction parity against the original Kaggle inference implementation, batch processing, and report generation. Run the app interactively after installation to check visual behavior and take a genuine screenshot for Figure 22.

## Scope

Research demonstration only; simulated virtual clients from one cohort, centrally estimated preprocessing, and no demonstrated clinical deployment or privacy guarantee. Not for diagnosis, screening, treatment, or clinical risk assessment. No medical record storage is implemented.

## Deployment to Streamlit Community Cloud

Push the files from this folder (including `models/`) to the GitHub repo. Set main file path to `app.py`; use `requirements.txt`. Do not point the app to the archived `legacy_reference/` weights or scalers. Refresh the hosted app and use a manual sample classification before capturing a manuscript screenshot.
