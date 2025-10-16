import os
import io
import pickle
import random
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF

# -----------------------------
# Feature Validation Data (from your Excel file)
# -----------------------------
VALIDATION_FEATURES = {
    "Age": {
        "description": "Older age increases ovarian cancer risk; clinicians consider it when interpreting markers.",
        "role": "Risk Factor",
        "link": ""
    },
    "Menopause": {
        "description": "Hormone changes after menopause affect marker interpretation; postmenopausal people have higher risk.",
        "role": "Hormonal Context",
        "link": "https://my.clevelandclinic.org/health/diseases/21841-menopause"
    },
    "HE4": {
        "description": "Blood protein used to detect or monitor ovarian cancer; high levels may suggest active disease.",
        "role": "Cancer Marker",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/62137#clinical-and-interpretive"
    },
    "CA125": {
        "description": "Tumor antigen for ovarian cancer; high values may indicate tumor activity, low values are reassuring.",
        "role": "Cancer Marker",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/9289#clinical-and-interpretive"
    },
    "CEA": {
        "description": "General cancer marker; high values may indicate malignancy, especially in certain ovarian types.",
        "role": "Cancer Marker (Non-Specific)",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/8521#clinical-and-interpretive"
    },
    "CA72-4": {
        "description": "Adjunct tumor marker; high levels can support cancer diagnosis but not definitive alone.",
        "role": "Adjunct Marker",
        "link": "https://mefact.org/blood-test-ca-72-4-what-you-need-to-know-m164.html"
    },
    "PCT": {
        "description": "Rises in bacterial infection; high values suggest bacterial cause, low values reduce likelihood.",
        "role": "Inflammation Marker",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/602598#clinical-and-interpretive"
    },
    "NEU": {
        "description": "Neutrophils rise in infection or stress; low counts suggest infection risk or marrow suppression.",
        "role": "Immune Marker",
        "link": "https://my.clevelandclinic.org/health/body/22313-neutrophils"
    },
    "LYM%": {
        "description": "Percentage of lymphocytes; high often shows immune activation, low suggests weak immunity.",
        "role": "Immune Marker",
        "link": "https://my.clevelandclinic.org/health/body/23342-lymphocytes"
    },
    "LYM#": {
        "description": "Absolute lymphocyte count; low indicates immune suppression, high indicates infection or activation.",
        "role": "Immune Marker",
        "link": "https://my.clevelandclinic.org/health/body/23342-lymphocytes"
    },
    "MONO#": {
        "description": "Monocytes reflect chronic inflammation or recovery; very low counts may indicate marrow issues.",
        "role": "Inflammation Marker",
        "link": "https://my.clevelandclinic.org/health/body/22110-monocytes"
    },
    "PLT": {
        "description": "Platelets control clotting; high counts can appear in inflammation or cancer, low counts increase bleeding risk.",
        "role": "Inflammation Marker",
        "link": "https://my.clevelandclinic.org/health/body/22879-platelets"
    },
    "MCH": {
        "description": "Average hemoglobin per red cell; low suggests iron-deficiency anemia, high suggests B12/folate issues.",
        "role": "Hematology",
        "link": "https://my.clevelandclinic.org/health/diagnostics/mch-blood-test"
    },
    "HGB": {
        "description": "Hemoglobin carries oxygen; low indicates anemia, high may reflect dehydration or lung disease.",
        "role": "Hematology",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/82080#clinical-and-interpretive"
    },
    "AST": {
        "description": "Enzyme from liver/heart/muscle; high levels indicate tissue injury or liver disease.",
        "role": "Liver Function",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/8360#clinical-and-interpretive"
    },
    "ALP": {
        "description": "Liver/bone enzyme; high ALP suggests bile obstruction or bone disease.",
        "role": "Liver/Bone Function",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/8340#clinical-and-interpretive"
    },
    "TBIL": {
        "description": "Total bilirubin; high values indicate hemolysis or poor liver/bile excretion.",
        "role": "Liver Function",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/81785#clinical-and-interpretive"
    },
    "DBIL": {
        "description": "Direct bilirubin; high values point to bile duct blockage or liver issues.",
        "role": "Liver Function",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/81787#clinical-and-interpretive"
    },
    "IBIL": {
        "description": "Indirect bilirubin; high levels suggest hemolysis or immature liver function.",
        "role": "Liver Function",
        "link": ""
    },
    "ALB": {
        "description": "Albumin shows nutrition/liver function; low levels indicate malnutrition or liver disease.",
        "role": "Nutrition Status",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/610525#clinical-and-interpretive"
    },
    "TP": {
        "description": "Total protein; low suggests malnutrition/liver disease, high may reflect chronic inflammation.",
        "role": "Nutrition Status",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/8520#clinical-and-interpretive"
    },
    "GLO": {
        "description": "Globulins are immune proteins; high suggests inflammation, low indicates low antibodies.",
        "role": "Immune/Protein Status",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/608102#clinical-and-interpretive"
    },
    "Na": {
        "description": "Sodium controls fluid balance; low causes weakness/confusion, high reflects dehydration.",
        "role": "Electrolyte Balance",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/602353#clinical-and-interpretive"
    },
    "Ca": {
        "description": "Calcium supports bones, nerves, and heart; low reflects deficiency, high may indicate overactive parathyroid.",
        "role": "Electrolyte Balance",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/601514#clinical-and-interpretive"
    },
    "GLU.": {
        "description": "Blood sugar level; high suggests diabetes/stress, low indicates hypoglycemia.",
        "role": "Metabolic Marker",
        "link": "https://www.mayocliniclabs.com/test-catalog/overview/89115#clinical-and-interpretive"
    }
}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def try_extract_input_feature_names(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for s in model.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            try:
                df2[c] = pd.to_numeric(df2[c])
            except Exception:
                pass
    return df2

def risk_label_from_proba(p_high: float) -> str:
    if p_high < 0.40:
        return "Low Risk"
    elif p_high < 0.70:
        return "Moderate Risk"
    else:
        return "High Risk"

def style_risk(v: str):
    if v == "Low Risk":
        return "background-color: #4CAF50; color: white;"
    elif v == "Moderate Risk":
        return "background-color: #FFEB3B; color: black;"
    else:
        return "background-color: #F44336; color: white;"

# -----------------------------
# SHAP explanation + Clinical Facts
# -----------------------------
def explain_with_shap(input_df, shap_values_df, feature_names, abs_threshold: float = 0.02, top_k: int = 8):
    if shap_values_df is None:
        return []

    cols = [c for c in feature_names if c in shap_values_df.columns]
    if not cols:
        return []

    mean_abs = shap_values_df[cols].abs().mean()
    mean_signed = shap_values_df[cols].mean()

    strong = mean_abs[mean_abs >= abs_threshold].sort_values(ascending=False)
    if strong.empty:
        strong = mean_abs.sort_values(ascending=False).head(min(top_k, len(cols)))

    table_data = []
    for feat in strong.index:
        if mean_signed[feat] > 0:
            val = float(input_df.iloc[0].get(feat, np.nan))
            feature_info = VALIDATION_FEATURES.get(feat, {
                "description": "Clinical monitoring and follow-up advised.",
                "role": "General Marker",
                "evidence_strength": "Low",
                "link": ""
            })

            table_data.append({
                "Feature": feat,
                "Value": f"{val:.2f}",
                "Risk": "High Risk",
                "Role": feature_info["role"],
                "Facts": feature_info["description"]
                
            })

    if not table_data:
        return []

    df_table = pd.DataFrame(table_data)
    df_table = df_table.astype(str)
    table_html = df_table.to_html(index=False, escape=False)
    styled_html = f"""
    <div style='overflow-x:auto;'>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left !important;
                vertical-align: top;
                white-space: normal;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
        </style>
        {table_html}
    </div>
    """
    return df_table, styled_html

# -----------------------------
# PDF Generation Function
# -----------------------------
def generate_pdf_report(user_vals, risk_label, percent, df_table=None):
    pdf = FPDF()
    pdf.add_page()
   
    # Set margins: left, top, right (in mm)
    pdf.set_left_margin(25)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(auto=True, margin=15)
   
    # Title
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "OvaPredictAI - Ovarian Cancer Prediction Report", ln=True, align="C")
    pdf.ln(3)
   
    # Patient Information Section - More compact
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Patient Input Values", ln=True)
    pdf.ln(3)
   
    # Input values table - 3 columns: Feature, Full Name, Value
    pdf.set_font("Arial", "B", 8)
    col_widths = [40, 80, 40]  # Feature, Full Name, Value
    pdf.cell(col_widths[0], 6, "Feature", border=1)
    pdf.cell(col_widths[1], 6, "Full Name", border=1)
    pdf.cell(col_widths[2], 6, "Value", border=1, align="R")
    pdf.ln()
   
    pdf.set_font("Arial", "", 8)
    for feature, value in user_vals.items():
        # Get full name for the feature
        full_name = get_feature_display_name(feature)
        # Truncate full name if too long
        full_name_display = full_name[:35] + "..." if len(full_name) > 38 else full_name
       
        pdf.cell(col_widths[0], 6, feature, border=1)
        pdf.cell(col_widths[1], 6, full_name_display, border=1)
        pdf.cell(col_widths[2], 6, f"{value:.2f}", border=1, align="R")
        pdf.ln()
   
    pdf.ln(3)
   
    # Prediction Result Section - More compact
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Prediction Result", ln=True)
    pdf.ln(3)
   
    pdf.set_font("Arial", "B", 14)
    if risk_label == "High Risk":
        pdf.set_text_color(255, 0, 0)
    elif risk_label == "Moderate Risk":
        pdf.set_text_color(255, 165, 0)
    else:
        pdf.set_text_color(0, 128, 0)
   
    pdf.cell(0, 8, f"Risk Assessment: {risk_label} ({percent:.2f}%)", ln=True)
    pdf.ln(3)
   
    # Risk Indicators Section (if any) - More compact
    if df_table is not None and not df_table.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "3. High-Risk Indicators & Clinical Facts", ln=True)
        pdf.ln(3)
       
        # More compact column widths for risk indicators table
        col_widths = [15, 15, 15, 20, 95]  # Feature, Value, Risk, Role, Facts
       
        # Table header - smaller font
        pdf.set_font("Arial", "B", 7)
        headers = ["Feature", "Value", "Risk", "Role", "Clinical Facts"]
       
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 6, header, border=1)
        pdf.ln()
       
        # Table rows - smaller font
        pdf.set_font("Arial", "", 7)
        for idx, row in df_table.iterrows():
            # Feature
            feature = str(row["Feature"])
            pdf.cell(col_widths[0], 6, feature, border=1)
           
            # Value
            value = str(row["Value"])
            pdf.cell(col_widths[1], 6, value, border=1)
           
            # Risk
            risk = str(row["Risk"])
            pdf.cell(col_widths[2], 6, risk, border=1)
           
            # Role
            role = str(row["Role"])
            pdf.cell(col_widths[3], 6, role, border=1)
           
            # Facts - multi_cell for wrapping with proper width
            facts = str(row["Facts"])
            # Calculate height needed for this cell
            text_width = col_widths[4] - 2  # Account for borders
            text_height = 3.0  # Smaller base height
           
            # Simple line break calculation
            words = facts.split()
            lines = []
            current_line = ""
           
            for word in words:
                test_line = current_line + word + " "
                # Approximate width calculation
                if pdf.get_string_width(test_line) < text_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
           
            # Calculate cell height
            cell_height = max(6, len(lines) * text_height)  # Minimum 6, adjust based on lines
           
            # Get current position
            x = pdf.get_x()
            y = pdf.get_y()
           
            # Draw border for Facts cell
            pdf.cell(col_widths[4], cell_height, "", border=1)  # Empty cell for border
            pdf.set_xy(x, y)  # Reset position to start of cell
           
            # Print text with line breaks
            for i, line in enumerate(lines):
                pdf.set_xy(x, y + (i * text_height))
                pdf.cell(col_widths[4], text_height, line, 0, 0, 'L')
           
            # Move to next row position
            pdf.set_xy(x + col_widths[4], y + cell_height)
            pdf.ln(cell_height - 6)  # Adjust line spacing
   
    # Compact footer
    pdf.ln(3)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, "Generated by OvaPredictAI - For clinical decision support only", ln=True, align="R")
   
    return pdf

# Feature display name mapping function
def get_feature_display_name(short_name):
    feature_mapping = {
        # Blood tests and biomarkers
        'Age': 'Age',
        'HE4': 'Human Epididymis Protein 4',
        'CA125': 'Cancer Antigen 125',
        'CA72-4': 'Cancer Antigen 72-4',
        'CEA': 'Carcinoembryonic Antigen',
       
        # Complete Blood Count (CBC) parameters
        'HGB': 'Hemoglobin',
        'PLT': 'Platelets',
        'NEU': 'Neutrophils',
        'LYM#': 'Lymphocyte Count',
        'LYM%': 'Lymphocyte Percentage',
        'MONO#': 'Monocyte Count',
        'PCT': 'Procalcitonin',
       
        # Liver function tests
        'ALB': 'Albumin',
        'ALP': 'Alkaline Phosphatase',
        'AST': 'Aspartate Aminotransferase',
        'TBIL': 'Total Bilirubin',
        'DBIL': 'Direct Bilirubin',
        'IBIL': 'Indirect Bilirubin',
        'TP': 'Total Protein',
        'GLO': 'Globulin',
       
        # Electrolytes and other tests
        'Na': 'Sodium',
        'Ca': 'Calcium',
        'GLU.': 'Glucose',
        'MCH': 'Mean Corpuscular Hemoglobin',
       
        # Clinical parameters
        'Menopause': 'Menopausal Status'
    }
   
    return feature_mapping.get(short_name, short_name)

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="OvaPredictAI", layout="wide")
st.title("OvaPredictAI: Ovarian Cancer Prediction")

with st.sidebar:
    st.header("Settings")
    default_model_path = "overall_best_federated_xgb.pkl"
    default_scaler_path = "scaler_hybrid.pkl"

    uploaded_model = st.file_uploader("Upload a .pkl model", type=["pkl"])
    uploaded_scaler = st.file_uploader("Upload a .pkl scaler", type=["pkl"])

    if uploaded_model:
        with open("uploaded_model.pkl", "wb") as f:
            f.write(uploaded_model.read())
        model_path = "uploaded_model.pkl"
    else:
        model_path = default_model_path

    try:
        model_obj = load_model(model_path)
        if isinstance(model_obj, list):
            st.warning("Model file contained a list. Using the first element.")
            model_obj = model_obj[0]
        st.success(f"Loaded model: {os.path.basename(model_path)}")
    except Exception as e:
        model_obj = None
        st.error(f"Could not load model: {e}")

    if uploaded_scaler:
        with open("uploaded_scaler.pkl", "wb") as f:
            f.write(uploaded_scaler.read())
        with open("uploaded_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        try:
            with open(default_scaler_path, "rb") as f:
                scaler = pickle.load(f)
            st.success("Scaler loaded")
        except Exception:
            scaler = None
            st.warning("No scaler found. Upload scaler_hybrid.pkl")

# -----------------------------
# Load dataset medians & SHAP
# -----------------------------
try:
    df = pd.read_csv("selected_features_data.csv")
    medians = df.median(numeric_only=True)
except Exception:
    medians = {}

try:
    shap_values_df = pd.read_csv("shap_values.csv")
    st.sidebar.success("Loaded SHAP values from shap_values.csv")
except Exception:
    shap_values_df = None
    st.sidebar.warning("No shap_values.csv found. Risk indicators will be limited.")

FALLBACK_FEATURE_NAMES = [
    'Age', 'HE4', 'Menopause', 'CA125', 'ALB', 'NEU', 'LYM%', 'ALP',
    'PLT', 'LYM#', 'AST', 'PCT', 'IBIL', 'TBIL', 'CA72-4', 'GLO',
    'MONO#', 'HGB', 'Na', 'CEA', 'Ca', 'GLU.', 'DBIL', 'TP', 'MCH'
]

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Single Prediction", "Batch Prediction", "Feature Reference"])

# -----------------------------
# Single Prediction
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction")

    if model_obj is None:
        st.info("Load a model from the sidebar to begin.")
    else:
        feature_names = try_extract_input_feature_names(model_obj) or FALLBACK_FEATURE_NAMES
        st.markdown("Enter the feature values:")

        if "static_defaults" not in st.session_state:
            st.session_state.static_defaults = {
                feat: float(medians.get(feat, random.uniform(1, 100)))
                for feat in feature_names
            }

        cols = st.columns(min(4, len(feature_names)))
        user_vals = {}

        for i, feat in enumerate(feature_names):
            with cols[i % len(cols)]:
                default_val = st.session_state.static_defaults[feat]
                if feat.lower() == "age":
                    user_input = st.number_input(
                        f"🔹 {feat}",
                        value=int(default_val),
                        step=1,
                        format="%d",
                        key=f"input_{feat}_{i}"
                    )
                else:
                    user_input = st.number_input(
                        f"🔹 {feat}",
                        value=float(default_val),
                        step=0.1,
                        format="%.3f",
                        key=f"input_{feat}_{i}"
                    )
                user_vals[feat] = float(user_input)

        if st.button("Predict", type="primary"):
            try:
                clean_vals = {k: float(v) for k, v in user_vals.items()}
                input_df = pd.DataFrame([clean_vals], columns=feature_names)

                if scaler:
                    X_scaled = scaler.transform(input_df)
                    X_df = pd.DataFrame(X_scaled, columns=feature_names, index=[0])
                else:
                    st.warning("Scaler not loaded. Using raw values.")
                    X_df = input_df

                proba = model_obj.predict_proba(X_df)[0]
                classes = getattr(model_obj, "classes_", np.array([0, 1]))
                idx_high = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
                p_high = float(proba[idx_high])

                risk_label = risk_label_from_proba(p_high)
                percent = p_high * 100

                # Display risk card
                risk_colors = {"Low Risk": "#4CAF50", "Moderate Risk": "#FFEB3B", "High Risk": "#F44336"}
                color = risk_colors.get(risk_label, "#000000")
                st.markdown(f"""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    background-color: {color};
                    color: {'white' if risk_label == 'High Risk' else 'black'};
                    font-size: 28px;
                    font-weight: bold;
                    text-align: center;
                    box-shadow: 2px 2px 12px rgba(0,0,0,0.2);
                    margin-bottom: 20px;
                ">{risk_label} ({percent:.2f}%)</div>
                """, unsafe_allow_html=True)

                # High risk indicators table
                st.markdown("### Risk Indicators")
                df_table = None
                if risk_label == "Low Risk":
                    st.markdown("✅ All features are within normal range.")
                else:
                    result = explain_with_shap(input_df, shap_values_df, feature_names)
                    if result and len(result) == 2:
                        df_table, table_html = result
                        if df_table is not None and not df_table.empty:
                            st.markdown(table_html, unsafe_allow_html=True)

                # Generate and display PDF download button (always available)
                st.markdown("### Download Complete Report")
                pdf = generate_pdf_report(user_vals, risk_label, percent, df_table)
               
                pdf_buffer = io.BytesIO()
                pdf.output(pdf_buffer)
                pdf_buffer.seek(0)

                st.download_button(
                    label="Download Full Report (PDF)",
                    data=pdf_buffer,
                    file_name="OvaPredictAI_complete_report.pdf",
                    mime="application/pdf",
                    type="primary"
                )

                # Download CSV
                if df_table is not None and not df_table.empty:
                    csv_buf = io.StringIO()
                    df_table.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="Download High-Risk Indicators (CSV)",
                        data=csv_buf.getvalue(),
                        file_name="OvaPredictAI_high_risk_indicators.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Batch Prediction
# -----------------------------
with tabs[1]:
    st.subheader("Batch Prediction")
    if model_obj:
        batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch")
        if batch_file:
            df = pd.read_csv(batch_file)
            st.write("Preview:", df.head())

            if st.button("Predict (Batch)"):
                try:
                    feature_names = try_extract_input_feature_names(model_obj) or FALLBACK_FEATURE_NAMES

                    if scaler is None:
                        st.error("Scaler not loaded! Upload scaler_hybrid.pkl")
                    else:
                        X_scaled = scaler.transform(df[feature_names])
                        df_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=df.index)

                        proba = model_obj.predict_proba(df_scaled)
                        classes = getattr(model_obj, "classes_", np.array([0, 1]))
                        idx_high = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
                        p_high = proba[:, idx_high]

                        result = pd.DataFrame(index=df.index)
                        result["High Risk (%)"] = (p_high * 100).round(2).astype(str) + "%"
                        result["Risk Level"] = [risk_label_from_proba(p) for p in p_high]

                        st.success("Predictions completed.")
                        st.dataframe(
                            result.head(20).style.applymap(lambda v: style_risk(v), subset=["Risk Level"]),
                            use_container_width=True
                        )

                        csv_buf = io.StringIO()
                        result.to_csv(csv_buf, index=False)
                        st.download_button("📄 Download Batch Predictions (CSV)", csv_buf.getvalue(), "batch_predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Prediction failed. Error: {e}")


# -----------------------------
# Tab 3: Clinical Interpretation (color-tinted grouped cards with expanded Mayo-style explanations)
# -----------------------------
# assume you have: tab1, tab2, tab3 = st.tabs([...])
with tabs[2]:
    st.header("Clinical Interpretation Reference")

    # Category definitions: features grouped and a soft color for each group
    categories = {
        "Risk Factors": {
            "features": ["Age", "Menopause"],
            "color": "#FFF4E6"  # pale peach
        },
        "Tumor Markers": {
            "features": [
                "Human Epididymis Protein 4 | HE4",
                "Cancer Antigen 125 | CA125",
                "Carcinoembryonic Antigen | CEA",
                "Cancer Antigen 72-4 | CA72-4"
            ],
            "color": "#E8F4FF"  # pale blue
        },
        "Blood & Immune Markers": {
            "features": [
                "Procalcitonin | PCT",
                "Neutrophils | NEU",
                "Lymphocyte Percentage | LYM%",
                "Lymphocyte Count | LYM#",
                "Monocyte Count | MONO#",
                "Platelets | PLT",
                "Mean Corpuscular Hemoglobin | MCH",
                "Hemoglobin | HGB"
            ],
            "color": "#E8FFF3"  # pale green
        },
        "Liver & Metabolic Markers": {
            "features": [
                "Aspartate Aminotransferase | AST",
                "Alkaline Phosphatase | ALP",
                "Total Bilirubin | TBIL",
                "Direct Bilirubin | DBIL",
                "Indirect Bilirubin | IBIL",
                "Albumin | ALB",
                "Total Protein | TP",
                "Globulin | GLO",
                "Glucose | GLU."
            ],
            "color": "#FFF1F2"  # pale pink
        },
        "Electrolytes & Nutrition": {
            "features": ["Sodium | Na", "Calcium | Ca"],
            "color": "#F5F5FF"  # pale lavender
        }
    }


    # Expanded 2-3 paragraph explanations (Mayo-style, simplified) for each feature.
    # We'll map brief expanded_texts derived from the Mayo-style summaries you approved.
    expanded_texts = {
        "Age": (
            "Age itself is not a lab test but a clinical factor; older age raises the chance of ovarian cancer. "
            "Most ovarian cancers occur in older adults, and risk increases after menopause. Clinicians always consider age when interpreting lab results and deciding on further testing or imaging."
        ),
        "Menopause": (
            "Menopause is the permanent end of menstrual cycles, identified after 12 months without a period. "
            "Hormone shifts after menopause change how markers behave; some algorithms use menopausal status to set different risk thresholds. Menopausal state also affects screening and follow-up strategies."
        ),
        "Human Epididymis Protein 4 | HE4": (
            "HE4 is a blood protein often higher in epithelial ovarian cancers. It's measured using a lab immunoassay. ",
            "HE4 is most useful to monitor patients who already had ovarian cancer; rising values can indicate recurrence, while falling values suggest response to treatment. It's often combined with CA125 for better accuracy.",
            "Higher-than-normal HE4 raises concern for ovarian tumor activity but must be interpreted with clinical context because some benign conditions may also alter levels."
        ),
        "Cancer Antigen 125 | CA125": (
            "CA125 is a protein frequently produced by ovarian cancer cells and measured in blood. ",
            "Clinicians commonly use CA125 to track response to therapy and to detect recurrence after treatment; it is not recommended alone for general population screening due to false positives.",
            "Steady rises in CA125 after treatment suggest residual disease or recurrence, while isolated mild increases may be from benign causes like menstruation or inflammation."
        ),
        "Carcinoembryonic Antigen | CEA": (
            "CEA is a tumor-associated protein used widely in colorectal cancer but sometimes raised in ovarian tumors (especially mucinous types).",
            "CEA is used alongside other markers to help determine tumor origin or monitor disease progression in cancers where it is informative.",
            "High CEA suggests malignancy or heavy inflammation and usually triggers follow-up imaging or additional tumor-specific tests."
        ),
        "Cancer Antigen 72-4 | CA72-4": (
            "CA72-4 is a tumor marker that may be elevated in ovarian or gastrointestinal cancers.",
            "It's considered an adjunct, helpful when used with CA125 and clinical findings, but rarely used as a single diagnostic test.",
            "Elevated CA72-4 supports further investigation but is not definitive; doctors combine it with imaging and other markers."
        ),
        "Procalcitonin | PCT": (
            "Procalcitonin (PCT) increases when the body has a serious bacterial infection.",
            "PCT helps differentiate bacterial infections from viral or inflammatory conditions and can guide antibiotic decisions in hospital and ICU settings.",
            "High PCT often indicates bacterial sepsis or severe infection; low values suggest bacterial infection is unlikely but don't rule out other causes."
        ),
        "Neutrophils | NEU": (
            "Neutrophils are white blood cells that respond rapidly to infection.",
            "Doctors use neutrophil counts to detect bacterial infection or inflammation and to track response to therapy.",
            "High neutrophils point to bacterial infection or inflammation; low neutrophils increase infection risk and require urgent evaluation."
        ),
        "Lymphocyte Percentage | LYM%": (
            " Lymphocyte percentage shows the proportion of white blood cells that are lymphocytes.",
            "It complements the absolute lymphocyte count to assess immune status and response to infections or immune conditions.",
            "A high lymphocyte percentage often accompanies viral infections or immune activation; low percentages suggest weakened immune function."
        ),
        "Lymphocyte Count | LYM#": (
            "Absolute lymphocyte count is the total number of lymphocytes in blood.",
            "It's used to evaluate immune strength, detect infections, and monitor conditions affecting lymphocytes (e.g., HIV or certain blood cancers).",
            "Low absolute lymphocytes point to immune suppression or bone marrow issues; high counts suggest infection or chronic immune activation."
        ),
        "Monocyte Count | MONO#": (
            "Monocytes are white blood cells involved in longer-term immune responses and tissue repair.",
            "Monocyte counts help suggest chronic infection, inflammation, or recovery phases after acute illness.",
            "Elevated monocytes often indicate chronic inflammation or certain infections; low counts usually reflect bone marrow suppression or acute illness."
        ),
        "Platelets | PLT": (
            "Platelets are small blood components that stop bleeding by forming clots.",
            "Platelet count is checked before surgery, during chemotherapy, and when bleeding or clotting problems are suspected.",
            "High platelets may be a sign of inflammation or malignancy risk; low platelets raise bleeding concerns and often trigger further evaluation."
        ),
        "Mean Corpuscular Hemoglobin | MCH": (
            "MCH estimates the average hemoglobin per red blood cell and helps classify types of anemia.",
            "MCH is used with other RBC indices (MCV, MCHC) to decide whether anemia is due to iron deficiency or vitamin B12/folate problems.",
            "Low MCH points toward iron-deficiency microcytic anemia; high MCH suggests macrocytic anemia from B12/folate deficiency."
        ),
        "Hemoglobin | HGB": (
            "Hemoglobin reflects the blood's oxygen-carrying capacity.",
            "HGB guides diagnosis of anemia, transfusion needs, and overall clinical fitness for procedures.",
            "Low HGB causes fatigue and signals anemia; high HGB may indicate dehydration or chronic lung disease."
        ),
        "Aspartate Aminotransferase | AST": (
            "AST is an enzyme released when liver or muscle cells are damaged.",
            "AST helps detect liver injury, monitor hepatitis, and assess muscle damage; it's used with ALT and other liver tests for context.",
            "High AST suggests liver or muscle injury; the degree and pattern relative to ALT help suggest specific causes (viral, alcoholic, or other)."
        ),
        "Alkaline Phosphatase | ALP": (
            "ALP is an enzyme from liver and bone that rises when bile flow is blocked or bone turnover is increased.",
            "ALP helps distinguish liver vs bone sources of disease when paired with other tests (e.g., GGT for liver).",
            "Very high ALP often points to bile duct obstruction or bone disease; moderate rises occur in many liver disorders."
        ),
        "Total Bilirubin | TBIL": (
            "Total bilirubin is the sum of processed and unprocessed bilirubin reflecting red cell breakdown and liver clearance.",
            "TBIL helps evaluate jaundice causes and overall liver excretory function.",
            "Elevated TBIL indicates hemolysis, impaired liver conjugation, or biliary obstruction and usually prompts imaging and additional liver testing."
        ),
        "Direct Bilirubin | DBIL": (
            "Direct (conjugated) bilirubin shows how well the liver processes and excretes bilirubin.",
            "High direct bilirubin focuses attention on obstructive or cholestatic liver disease and biliary problems.",
            "Elevated DBIL suggests obstruction of bile flow or hepatocellular excretion problems, often requiring imaging or specialist referral."
        ),
        "Indirect Bilirubin | IBIL": (
            "Indirect (unconjugated) bilirubin indicates how much bilirubin is circulating before liver processing.",
            "High indirect bilirubin raises concern for hemolysis or inherited conjugation disorders; it's evaluated alongside TBIL/DBIL.",
            "Elevated IBIL usually points to red cell breakdown (hemolysis) or reduced liver conjugation capacity."
        ),
        "Albumin | ALB": (
            "Albumin is the main blood protein reflecting nutrition and liver synthetic ability.",
            "Low albumin signals poor nutrition, chronic inflammation, or impaired liver function and affects drug dosing and prognosis.",
            "Markedly low albumin often indicates severe disease or malnutrition; high albumin commonly reflects dehydration."
        ),
        "Total Protein | TP": (
            "Total protein measures albumin plus globulins and gives a broad view of nutritional and immune status.",
            "TP helps detect protein-losing states, liver disease, or high protein production from immune disorders.",
            "Low TP points to malnutrition or protein loss; high TP suggests chronic inflammation or plasma cell disorders and needs further testing."
        ),
        "Globulin | GLO": (
            "Globulins include antibodies and other immune proteins produced in infections or immune activation.",
            "Elevated globulins prompt evaluation for chronic infections, autoimmune disease, or disorders like multiple myeloma.",
            "High globulin levels indicate immune activation or chronic inflammation; low levels suggest poor antibody production."
        ), 
        "Glucose | GLU.": (
            "Blood glucose indicates current blood sugar levels.",
            "Glucose testing diagnoses diabetes and guides immediate management of hypo- or hyperglycemia.",
            "High glucose points to diabetes or stress hyperglycemia; low glucose can cause dizziness, weakness, and requires urgent correction."
        ),
        "Sodium | Na": (
            "Sodium shows how well the body balances water and electrolytes.",
            "Sodium helps diagnose dehydration, fluid overload, or hormone imbalances and guides fluid therapy.",
            "Low sodium causes confusion and seizures in severe cases; high sodium usually reflects water loss and requires careful correction."
        ),
        "Calcium | Ca": (
            "Calcium supports bone health, nerve signaling, and muscle function.",
            "Calcium testing evaluates bone disease, parathyroid function, and some cancers that raise calcium.",
            "High calcium can cause nausea and arrhythmias and often stems from hyperparathyroidism or malignancy; low calcium causes muscle cramps and tetany."
        )
    }
    

        # render groups with color-tinted cards
        # render groups with color-tinted cards
    for cat_name, cat_info in categories.items():
        st.subheader(cat_name)
        features = cat_info["features"]
        bg = cat_info["color"]

        for feat in features:
            # get the expanded text paragraphs (some entries are tuples; join with paragraph breaks)
            texts = expanded_texts.get(feat, "")
            if isinstance(texts, tuple):
                paragraphs = [p for p in texts if p and p.strip()]
            else:
                paragraphs = [texts] if texts else ["No detailed summary available."]

            # extract abbreviation for lookup
            abbr = feat.split("|")[-1].strip()
            link = VALIDATION_FEATURES.get(abbr, {}).get("link", "")

            # card HTML
            link_html = f'<a href="{link}" target="_blank" style="text-decoration:none; color:#0056b3; font-weight:600;">Learn More</a>' if link else ""
            card_html = f"""
            <div style="background:{bg}; border:1px solid rgba(0,0,0,0.06); border-radius:12px; padding:16px; margin-bottom:16px;">
                <h4 style="margin:0 0 8px 0; color:#102a43;">{feat}</h4>
            """
            for p in paragraphs:
                card_html += f'<p style="margin:6px 0; color:#243b53; font-size:15px; line-height:1.5;">{p}</p>'
            card_html += f'<div style="margin-top:10px; text-align:right;">{link_html}</div></div>'

            st.markdown(card_html, unsafe_allow_html=True)

        st.write("---")



