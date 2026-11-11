"""OvaPredict AI Streamlit research prototype; V7 inference-bundle aligned."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
from research_inference import (load_bundle, raw_feature_names, selected_feature_names,
                                example_defaults, research_federated_predict,
                                make_report_pdf, validate_raw_data)

st.set_page_config(page_title="OvaPredict AI", page_icon="🧬", layout="wide")
VALIDATION_FEATURES = {
    "Age": {"description": "Older age increases ovarian cancer risk; clinicians consider it when interpreting markers.", "role": "Risk Factor", "link": ""},
    "Menopause": {"description": "Hormone changes after menopause affect marker interpretation; postmenopausal people have higher risk.", "role": "Hormonal", "link": "https://my.clevelandclinic.org/health/diseases/21841-menopause"},
    "HE4": {"description": "HE4 is a blood protein, used to detect or monitor ovarian cancer; high levels may suggest active disease.", "role": "Cancer Marker", "link": "https://www.mayocliniclabs.com/test-catalog/overview/62137#clinical-and-interpretive"},
    "CA125": {"description": "CA125 is a tumor marker associated with ovarian cancer; elevated levels may suggest active disease; lower levels are generally reassuring.", "role": "Cancer Marker", "link": "https://www.mayocliniclabs.com/test-catalog/overview/9289#clinical-and-interpretive"},
    "CEA": {"description": "General cancer marker; high values may indicate malignancy, especially in certain ovarian types.", "role": "Cancer Marker (Non-Specific)", "link": "https://www.mayocliniclabs.com/test-catalog/overview/8521#clinical-and-interpretive"},
    "CA72-4": {"description": "CA72-4 is a adjunct cancer marker; high levels can support cancer diagnosis but not definitive alone.", "role": "Cancer Marker", "link": "https://mefact.org/blood-test-ca-72-4-what-you-need-to-know-m164.html"},
    "PCT": {"description": "Procalcitonin rises in bacterial infection; high values suggest bacterial cause, low values reduce likelihood.", "role": "Inflammation Marker", "link": "https://www.mayocliniclabs.com/test-catalog/overview/602598#clinical-and-interpretive"},
    "NEU": {"description": "Neutrophils rise in infection or stress; low counts suggest infection risk or marrow suppression.", "role": "Immune Marker", "link": "https://my.clevelandclinic.org/health/body/22313-neutrophils"},
    "LYM%": {"description": "Percentage of lymphocytes; high often shows immune activation, low suggests weak immunity.", "role": "Immune Marker", "link": "https://my.clevelandclinic.org/health/body/23342-lymphocytes"},
    "LYM#": {"description": "Absolute lymphocyte count; low indicates immune suppression, high indicates infection or activation.", "role": "Immune Marker", "link": "https://my.clevelandclinic.org/health/body/23342-lymphocytes"},
    "MONO#": {"description": "Monocytes reflect chronic inflammation or recovery; very low counts may indicate marrow issues.", "role": "Inflammation Marker", "link": "https://my.clevelandclinic.org/health/body/22110-monocytes"},
    "PLT": {"description": "Platelets control clotting; high counts can appear in inflammation or cancer, low counts increase bleeding risk.", "role": "Inflammation Marker", "link": "https://my.clevelandclinic.org/health/body/22879-platelets"},
    "MCH": {"description": "Average hemoglobin per red cell; low suggests iron-deficiency anemia, high suggests B12/folate issues.", "role": "Hematology", "link": "https://my.clevelandclinic.org/health/diagnostics/mch-blood-test"},
    "HGB": {"description": "Hemoglobin carries oxygen; low indicates anemia, high may reflect dehydration or lung disease.", "role": "Hematology", "link": "https://www.mayocliniclabs.com/test-catalog/overview/82080#clinical-and-interpretive"},
    "AST": {"description": "Enzyme from liver/heart/muscle; high levels indicate tissue injury or liver disease.", "role": "Liver Function", "link": "https://www.mayocliniclabs.com/test-catalog/overview/8360#clinical-and-interpretive"},
    "ALP": {"description": "Aspartate Aminotransferase; Liver/bone enzyme; high ALP suggests bile obstruction or bone disease.", "role": "Liver Function", "link": "https://www.mayocliniclabs.com/test-catalog/overview/8340#clinical-and-interpretive"},
    "TBIL": {"description": "Total bilirubin; high values indicate hemolysis or poor liver/bile excretion.", "role": "Liver Function", "link": "https://www.mayocliniclabs.com/test-catalog/overview/81785#clinical-and-interpretive"},
    "DBIL": {"description": "Direct bilirubin; high values point to bile duct blockage or liver issues.", "role": "Liver Function", "link": "https://www.mayocliniclabs.com/test-catalog/overview/81787#clinical-and-interpretive"},
    "IBIL": {"description": "Indirect bilirubin; high levels suggest hemolysis or immature liver function.", "role": "Liver Function", "link": ""},
    "ALB": {"description": "Albumin shows nutrition/liver function; low levels indicate malnutrition or liver disease.", "role": "Nutrition Status", "link": "https://www.mayocliniclabs.com/test-catalog/overview/610525#clinical-and-interpretive"},
    "TP": {"description": "Total protein; low suggests malnutrition/liver disease, high may reflect chronic inflammation.", "role": "Nutrition Status", "link": "https://www.mayocliniclabs.com/test-catalog/overview/8520#clinical-and-interpretive"},
    "GLO": {"description": "Globulins are immune proteins; high suggests inflammation, low indicates low antibodies.", "role": "Immune/Protein Status", "link": "https://www.mayocliniclabs.com/test-catalog/overview/608102#clinical-and-interpretive"},
    "Na": {"description": "Sodium controls fluid balance; low causes weakness/confusion, high reflects dehydration.", "role": "Electrolyte Balance", "link": "https://www.mayocliniclabs.com/test-catalog/overview/602353#clinical-and-interpretive"},
    "Ca": {"description": "Calcium supports bones, nerves, and heart; low reflects deficiency, high may indicate overactive parathyroid.", "role": "Electrolyte Balance", "link": "https://www.mayocliniclabs.com/test-catalog/overview/601514#clinical-and-interpretive"},
    "GLU.": {"description": "Glucose shows blood sugar level; high suggests diabetes/stress, low indicates hypoglycemia.", "role": "Metabolic Marker", "link": "https://www.mayocliniclabs.com/test-catalog/overview/89115#clinical-and-interpretive"}
}


# The existing app's informational feature descriptions are retained below.

@st.cache_resource(show_spinner=False)
def active_bundle():
    return load_bundle()

st.title("OvaPredict AI: Ovarian Cancer Prediction")
st.caption("Research prototype | Simulated federated RFF-hinge SVM")
st.info("For research demonstration only. Predictions are experimental model classifications, not a clinical diagnosis or validated risk probability.")

try:
    bundle = active_bundle()
    all_features = raw_feature_names(bundle)
    chosen_features = selected_feature_names(bundle)
    defaults = example_defaults(bundle)
except Exception as exc:
    st.error(f"Unable to load the verified research model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Research Model")
    st.success("Verified federated SVM bundle loaded")
    st.caption(f"Inputs: {len(all_features)} original variables | {len(chosen_features)} selected features")
    st.caption("Simulation: 3 virtual clients; preprocessing fitted on pooled development data")
    st.caption(f"Standardized margin threshold: {float(bundle['standardized_threshold']):+.6f}")
    st.divider()
    st.warning("Do not enter identifiable patient information. No input records are saved by this app.")
    st.caption("The binary model output is based on a saved decision margin, not an estimated cancer probability.")

primary = [f for f in chosen_features if f in all_features]
additional = [f for f in all_features if f not in primary]

def render_feature_inputs(names, prefix):
    values = {}
    columns = st.columns(4)
    for index, feat in enumerate(names):
        with columns[index % 4]:
            if feat == 'Menopause':
                default = int(round(defaults.get(feat, 0)))
                val = st.selectbox('Menopause (0 = No, 1 = Yes)', [0, 1], index=max(0, min(1, default)), key=f'{prefix}_{feat}')
            else:
                med = defaults.get(feat, float('nan'))
                if not np.isfinite(med):
                    med = 0.0
                if feat == 'Age':
                    val = st.number_input(feat, value=float(round(med)), step=1.0, format='%.0f', key=f'{prefix}_{feat}')
                else:
                    val = st.number_input(feat, value=float(med), step=0.01, format='%.3f', key=f'{prefix}_{feat}')
            values[feat] = float(val)
    return values

def display_prediction(prediction):
    row = prediction.iloc[0]
    st.divider()
    a,b,c = st.columns(3)
    a.metric('Model classification', str(row['Model Class']))
    b.metric('Standardized decision score', f"{float(row['Standardized Decision Score']):+.4f}")
    c.metric('Decision threshold', f"{float(row['Decision Threshold']):+.4f}")
    st.caption('Class 1 = malignant, Class 0 = benign. Score is a standardized SVM decision margin (not a probability).')
    st.info('Research-only output. No clinically validated low/moderate/high risk bands are assigned.')

tabs = st.tabs(['Single Prediction', 'Batch Prediction', 'Feature Reference'])
with tabs[0]:
    st.subheader('Single Prediction')
    st.caption('Training-set medians are prefilled as example values for interface demonstration only. Replace them with complete measurements for an individual research record.')
    with st.form('single_prediction_form'):
        st.markdown('**Selected model features**')
        primary_values = render_feature_inputs([f for f in primary if f != 'CA72-4'], 'a')
        if 'CA72-4' in primary:
            with st.expander('CA72-4 measurement', expanded=True):
                ca_missing = st.checkbox('CA72-4 was not measured (use saved iterative imputer)', value=True)
                ca_value = st.number_input('CA72-4', value=0.0, min_value=0.0, step=0.01, format='%.3f', disabled=ca_missing)
                primary_values['CA72-4'] = np.nan if ca_missing else float(ca_value)
        with st.expander(f'Additional source measurements ({len(additional)})', expanded=False):
            st.caption('The saved imputer was fitted using all 49 original predictors. These additional measurements are part of the original preprocessing pipeline, even though the final classifier selects 25 features.')
            other_values = render_feature_inputs(additional, 'b')
        submitted = st.form_submit_button('Predict', type='primary')
    if submitted:
        try:
            user_vals = {**primary_values, **other_values}
            df = pd.DataFrame([{f: user_vals[f] for f in all_features}])
            result = research_federated_predict(df, bundle)
            st.session_state['last_single_input'] = df
            st.session_state['last_single_result'] = result
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')
    if 'last_single_result' in st.session_state:
        display_prediction(st.session_state['last_single_result'])
        try:
            report = make_report_pdf(st.session_state['last_single_input'], st.session_state['last_single_result'])
            st.download_button('Download research report (PDF)', data=report, file_name='OvaPredict_Research_Report.pdf', mime='application/pdf')
        except Exception as exc:
            st.error(f'Unable to prepare report: {exc}')

with tabs[1]:
    st.subheader('Batch Prediction')
    st.caption('Provide one row per record with the 49 original input column names. CA72-4 values can be empty for imputation. Other missing values use the fitted development-set imputer.')
    example = pd.DataFrame([{f: defaults[f] for f in all_features}])
    st.download_button('Download CSV template (demonstration medians)', data=example.to_csv(index=False).encode('utf-8'), file_name='OvaPredict_Batch_Template.csv', mime='text/csv')
    batch_file = st.file_uploader('Upload research data (CSV)', type=['csv'], key='batch_file')
    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file)
            st.caption(f'Records loaded: {len(df_batch)}')
            st.dataframe(df_batch.head(), use_container_width=True)
            checked = validate_raw_data(df_batch, bundle)
            if checked.isna().any().any():
                st.warning('Some measurements are missing. The saved development-fitted imputation rules will be applied; review missingness before interpreting outputs.')
            if st.button('Predict (Batch)', type='primary'):
                result = research_federated_predict(df_batch, bundle)
                export = df_batch.copy()
                for col in result.columns:
                    export[col] = result[col].to_numpy()
                st.success(f'Classified {len(export)} research records.')
                st.dataframe(export[['Model Class','Class Code','Standardized Decision Score','Decision Threshold']].head(20), use_container_width=True)
                st.download_button('Download batch predictions (CSV)', data=export.to_csv(index=False).encode('utf-8'), file_name='OvaPredict_Batch_Predictions.csv', mime='text/csv')
        except Exception as exc:
            st.error(f'Cannot process batch file: {exc}')

with tabs[2]:
    st.header("Feature Reference")
    st.caption("General information about measured variables. This reference is independent of the federated model decision and is not individual medical advice.")

    categories = {
        "Risk Factors": {
            "features": ["Age", "Menopause"],
            "color": "#FFF4E6"
        },
        "Tumor Markers": {
            "features": [
                "Human Epididymis Protein 4 | HE4",
                "Cancer Antigen 125 | CA125",
                "Carcinoembryonic Antigen | CEA",
                "Cancer Antigen 72-4 | CA72-4"
            ],
            "color": "#E8F4FF"
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
                "Hemoglobin | HGB",
                "Platelet Distribution Width | PDW",
                "Eosinophil Percentage | EO%"
            ],
            "color": "#E8FFF3"
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
            "color": "#FFF1F2"
        },
        "Electrolytes & Nutrition": {
            "features": ["Sodium | Na", "Calcium | Ca"],
            "color": "#F5F5FF"
        }
    }

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
            "HE4 is a blood protein often higher in epithelial ovarian cancers. It's measured using a lab immunoassay. "
            "HE4 is most useful to monitor patients who already had ovarian cancer; rising values can indicate recurrence, while falling values suggest response to treatment. It's often combined with CA125 for better accuracy. "
            "Higher-than-normal HE4 raises concern for ovarian tumor activity but must be interpreted with clinical context because some benign conditions may also alter levels."
        ),
        "Cancer Antigen 125 | CA125": (
            "CA125 is a protein frequently produced by ovarian cancer cells and measured in blood. "
            "Clinicians commonly use CA125 to track response to therapy and to detect recurrence after treatment; it is not recommended alone for general population screening due to false positives. "
            "Steady rises in CA125 after treatment suggest residual disease or recurrence, while isolated mild increases may be from benign causes like menstruation or inflammation."
        ),
        "Carcinoembryonic Antigen | CEA": (
            "CEA is a tumor-associated protein used widely in colorectal cancer but sometimes raised in ovarian tumors (especially mucinous types). "
            "CEA is used alongside other markers to help determine tumor origin or monitor disease progression in cancers where it is informative. "
            "High CEA suggests malignancy or heavy inflammation and usually triggers follow-up imaging or additional tumor-specific tests."
        ),
        "Cancer Antigen 72-4 | CA72-4": (
            "CA72-4 is a tumor marker that may be elevated in ovarian or gastrointestinal cancers. "
            "It's considered an adjunct, helpful when used with CA125 and clinical findings, but rarely used as a single diagnostic test. "
            "Elevated CA72-4 supports further investigation but is not definitive; doctors combine it with imaging and other markers."
        ),
        "Procalcitonin | PCT": (
            "Procalcitonin (PCT) increases when the body has a serious bacterial infection. "
            "PCT helps differentiate bacterial infections from viral or inflammatory conditions and can guide antibiotic decisions in hospital and ICU settings. "
            "High PCT often indicates bacterial sepsis or severe infection; low values suggest bacterial infection is unlikely but don't rule out other causes."
        ),
        "Neutrophils | NEU": (
            "Neutrophils are white blood cells that respond rapidly to infection. "
            "Doctors use neutrophil counts to detect bacterial infection or inflammation and to track response to therapy. "
            "High neutrophils point to bacterial infection or inflammation; low neutrophils increase infection risk and require urgent evaluation."
        ),
        "Lymphocyte Percentage | LYM%": (
            "Lymphocyte percentage shows the proportion of white blood cells that are lymphocytes. "
            "It complements the absolute lymphocyte count to assess immune status and response to infections or immune conditions. "
            "A high lymphocyte percentage often accompanies viral infections or immune activation; low percentages suggest weakened immune function."
        ),
        "Lymphocyte Count | LYM#": (
            "Absolute lymphocyte count is the total number of lymphocytes in blood. "
            "It's used to evaluate immune strength, detect infections, and monitor conditions affecting lymphocytes (e.g., HIV or certain blood cancers). "
            "Low absolute lymphocytes point to immune suppression or bone marrow issues; high counts suggest infection or chronic immune activation."
        ),
        "Monocyte Count | MONO#": (
            "Monocytes are white blood cells involved in longer-term immune responses and tissue repair. "
            "Monocyte counts help suggest chronic infection, inflammation, or recovery phases after acute illness. "
            "Elevated monocytes often indicate chronic inflammation or certain infections; low counts usually reflect bone marrow suppression or acute illness."
        ),
        "Platelets | PLT": (
            "Platelets are small blood components that stop bleeding by forming clots. "
            "Platelet count is checked before surgery, during chemotherapy, and when bleeding or clotting problems are suspected. "
            "High platelets may be a sign of inflammation or malignancy risk; low platelets raise bleeding concerns and often trigger further evaluation."
        ),
        "Platelet Distribution Width | PDW": (
            "PDW measures how much platelets vary in size. Normal values are roughly 9–17%. "
            "Higher PDW suggests increased platelet turnover, which can accompany inflammation, infection, or malignancy. "
            "PDW is a supportive marker, not diagnostic on its own; it is interpreted alongside platelet count and clinical context."
        ),
        "Eosinophil Percentage | EO%": (
            "Eosinophils are a small subset of white blood cells (usually <5% of the differential). "
            "They rise in allergic reactions and parasitic infections, and fall with stress, corticosteroids, or acute infection. "
            "Extremely elevated EO% (well above 7–10%) is unusual and warrants evaluation for allergy, parasitosis, or hematologic disorders."
        ),
        "Mean Corpuscular Hemoglobin | MCH": (
            "MCH estimates the average hemoglobin per red blood cell and helps classify types of anemia. "
            "MCH is used with other RBC indices (MCV, MCHC) to decide whether anemia is due to iron deficiency or vitamin B12/folate problems. "
            "Low MCH points toward iron-deficiency microcytic anemia; high MCH suggests macrocytic anemia from B12/folate deficiency."
        ),
        "Hemoglobin | HGB": (
            "Hemoglobin reflects the blood's oxygen-carrying capacity. "
            "HGB guides diagnosis of anemia, transfusion needs, and overall clinical fitness for procedures. "
            "Low HGB causes fatigue and signals anemia; high HGB may indicate dehydration or chronic lung disease."
        ),
        "Aspartate Aminotransferase | AST": (
            "AST is an enzyme released when liver or muscle cells are damaged. "
            "AST helps detect liver injury, monitor hepatitis, and assess muscle damage; it's used with ALT and other liver tests for context. "
            "High AST suggests liver or muscle injury; the degree and pattern relative to ALT help suggest specific causes (viral, alcoholic, or other)."
        ),
        "Alkaline Phosphatase | ALP": (
            "ALP is an enzyme from liver and bone that rises when bile flow is blocked or bone turnover is increased. "
            "ALP helps distinguish liver vs bone sources of disease when paired with other tests (e.g., GGT for liver). "
            "Very high ALP often points to bile duct obstruction or bone disease; moderate rises occur in many liver disorders."
        ),
        "Total Bilirubin | TBIL": (
            "Total bilirubin is the sum of processed and unprocessed bilirubin reflecting red cell breakdown and liver clearance. "
            "TBIL helps evaluate jaundice causes and overall liver excretory function. "
            "Elevated TBIL indicates hemolysis, impaired liver conjugation, or biliary obstruction and usually prompts imaging and additional liver testing."
        ),
        "Direct Bilirubin | DBIL": (
            "Direct (conjugated) bilirubin shows how well the liver processes and excretes bilirubin. "
            "High direct bilirubin focuses attention on obstructive or cholestatic liver disease and biliary problems. "
            "Elevated DBIL suggests obstruction of bile flow or hepatocellular excretion problems, often requiring imaging or specialist referral."
        ),
        "Indirect Bilirubin | IBIL": (
            "Indirect (unconjugated) bilirubin indicates how much bilirubin is circulating before liver processing. "
            "High indirect bilirubin raises concern for hemolysis or inherited conjugation disorders; it's evaluated alongside TBIL/DBIL. "
            "Elevated IBIL usually points to red cell breakdown (hemolysis) or reduced liver conjugation capacity."
        ),
        "Albumin | ALB": (
            "Albumin is the main blood protein reflecting nutrition and liver synthetic ability. "
            "Low albumin signals poor nutrition, chronic inflammation, or impaired liver function and affects drug dosing and prognosis. "
            "Markedly low albumin often indicates severe disease or malnutrition; high albumin commonly reflects dehydration."
        ),
        "Total Protein | TP": (
            "Total protein measures albumin plus globulins and gives a broad view of nutritional and immune status. "
            "TP helps detect protein-losing states, liver disease, or high protein production from immune disorders. "
            "Low TP points to malnutrition or protein loss; high TP suggests chronic inflammation or plasma cell disorders and needs further testing."
        ),
        "Globulin | GLO": (
            "Globulins include antibodies and other immune proteins produced in infections or immune activation. "
            "Elevated globulins prompt evaluation for chronic infections, autoimmune disease, or disorders like multiple myeloma. "
            "High globulin levels indicate immune activation or chronic inflammation; low levels suggest poor antibody production."
        ),
        "Glucose | GLU.": (
            "Blood glucose indicates current blood sugar levels. "
            "Glucose testing diagnoses diabetes and guides immediate management of hypo- or hyperglycemia. "
            "High glucose points to diabetes or stress hyperglycemia; low glucose can cause dizziness, weakness, and requires urgent correction."
        ),
        "Sodium | Na": (
            "Sodium shows how well the body balances water and electrolytes. "
            "Sodium helps diagnose dehydration, fluid overload, or hormone imbalances and guides fluid therapy. "
            "Low sodium causes confusion and seizures in severe cases; high sodium usually reflects water loss and requires careful correction."
        ),
        "Calcium | Ca": (
            "Calcium supports bone health, nerve signaling, and muscle function. "
            "Calcium testing evaluates bone disease, parathyroid function, and some cancers that raise calcium. "
            "High calcium can cause nausea and arrhythmias and often stems from hyperparathyroidism or malignancy; low calcium causes muscle cramps and tetany."
        )
    }

    for cat_name, cat_info in categories.items():
        st.subheader(cat_name)
        for feat in cat_info["features"]:
            texts = expanded_texts.get(feat, "")
            if isinstance(texts, tuple):
                paragraphs = [p for p in texts if p and p.strip()]
            else:
                paragraphs = [texts] if texts else ["No detailed summary available."]

            abbr = feat.split("|")[-1].strip()
            link = VALIDATION_FEATURES.get(abbr, {}).get("link", "")
            link_html = f'<a href="{link}" target="_blank" style="text-decoration:none; color:#0056b3; font-weight:600;">Learn More</a>' if link else ""
            card_html = f"""
            <div style="background:{cat_info['color']}; border:1px solid rgba(0,0,0,0.06); border-radius:12px; padding:16px; margin-bottom:16px;">
                <h4 style="margin:0 0 8px 0; color:#102a43;">{feat}</h4>
            """
            for p in paragraphs:
                card_html += f'<p style="margin:6px 0; color:#243b53; font-size:15px; line-height:1.5;">{p}</p>'
            card_html += f'<div style="margin-top:10px; text-align:right;">{link_html}</div></div>'
            st.markdown(card_html, unsafe_allow_html=True)
        st.write("---")
