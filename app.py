"""Streamlit demo for churn risk, local SHAP explanations, and retention actions."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.retention_assistant import fallback_recommendation, generate_recommendation


PROJECT_ROOT = Path(__file__).resolve().parent
GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "telco_gold.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Churn Retention Assistant", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction + Retention Assistant")
st.caption("Demo using the public IBM Telco Customer Churn sample dataset.")


@st.cache_data
def load_gold() -> pd.DataFrame:
    return pd.read_csv(GOLD_PATH)


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def get_api_key() -> str | None:
    try:
        return st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except Exception:
        # No secrets.toml is expected until the user adds their own local key.
        return os.getenv("GEMINI_API_KEY")


def get_local_explanation(pipeline, customer: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    transformed = preprocessor.transform(customer)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(transformed, check_additivity=False)
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    raw_values = np.asarray(raw_values)
    if raw_values.ndim == 3:
        values = raw_values[0, :, -1]
    elif raw_values.ndim == 2:
        values = raw_values[0]
    else:
        values = raw_values.reshape(-1)

    order = np.argsort(np.abs(values))[::-1][:8]
    reasons = []
    for index in order:
        name = str(feature_names[index])
        value = float(values[index])
        direction = "toward churn" if value > 0 else "away from churn"
        reasons.append({"feature": name, "shap_value": value, "direction": direction})
    return reasons


if not GOLD_PATH.exists() or not MODEL_PATH.exists():
    st.error("Project data or fitted model is missing. Run the data preparation and training scripts first.")
    st.stop()

gold = load_gold()
pipeline = load_pipeline()
features = gold.drop(columns="Churn")

with st.sidebar:
    st.header("Demo controls")
    record_number = st.number_input(
        "Sample customer number",
        min_value=1,
        max_value=len(features),
        value=1,
        step=1,
        help="This is a row number for the public sample, not a real customer ID.",
    )
    risk_threshold = st.slider(
        "High-risk score cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        format="%.2f",
    )

customer = features.iloc[[int(record_number) - 1]]
probability = float(pipeline.predict_proba(customer)[0, 1])
is_high_risk = probability >= risk_threshold

score_col, status_col = st.columns([1, 2])
with score_col:
    st.metric("Model churn score", f"{probability:.1%}")
with status_col:
    st.subheader("High risk" if is_high_risk else "Below selected cutoff")
    st.write(f"Selected cutoff: {risk_threshold:.0%}")
    st.caption("This is a model score; probability calibration has not been evaluated yet.")

st.subheader("Customer profile")
profile_cols = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
]
profile = customer.iloc[0]
visible_cols = st.columns(4)
for column_index, feature_name in enumerate(profile_cols):
    if feature_name in customer.columns:
        with visible_cols[column_index % len(visible_cols)]:
            value = profile[feature_name]
            if feature_name == "MonthlyCharges":
                st.metric("Monthly charges", f"${float(value):.2f}")
            else:
                st.metric(feature_name, str(value))

st.subheader("Why the model gave this score")
try:
    reasons = get_local_explanation(pipeline, customer)
    display_reasons = pd.DataFrame(reasons)
    display_reasons["impact"] = display_reasons["shap_value"].map(
        lambda value: "Toward churn" if value > 0 else "Away from churn"
    )
    display_reasons["shap_value"] = display_reasons["shap_value"].map(lambda value: f"{value:+.3f}")
    st.dataframe(
        display_reasons[["feature", "impact", "shap_value"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("SHAP values show direction and strength in the model's raw-score space, not percentage points.")
except Exception as error:
    reasons = []
    st.warning(f"Could not calculate this SHAP explanation: {error}")

st.subheader("Retention action")
if st.button("Generate retention recommendation", type="primary"):
    st.session_state.pop("gemini_error", None)
    customer_context = customer.iloc[0].to_dict()
    # The target label is not included in the customer profile sent to the LLM.
    customer_context.pop("Churn", None)
    with st.spinner("Choosing an action from the approved list…"):
        try:
            recommendation = generate_recommendation(
                customer_context,
                reasons,
                api_key=get_api_key(),
            )
        except Exception as error:
            api_key = get_api_key()
            safe_error = str(error)
            if api_key:
                safe_error = safe_error.replace(api_key, "[REDACTED API KEY]")
            recommendation = fallback_recommendation(customer_context)
            recommendation["rationale"] += f" Gemini request failed ({type(error).__name__}); local fallback used."
            st.session_state["gemini_error"] = safe_error
        st.session_state["recommendation"] = {
            "record_number": int(record_number),
            "result": recommendation,
        }

saved_recommendation = st.session_state.get("recommendation")
if saved_recommendation and saved_recommendation["record_number"] == int(record_number):
    recommendation = saved_recommendation["result"]
    st.markdown(f"**Suggested action:** {recommendation['action_name']}")
    st.write(recommendation["recommendation"])
    if recommendation.get("rationale"):
        st.caption(recommendation["rationale"])
    if recommendation.get("customer_message"):
        st.info(f"Suggested wording: {recommendation['customer_message']}")
    st.caption(f"Recommendation source: {recommendation['source']}")
    if saved_recommendation and st.session_state.get("gemini_error"):
        st.warning(f"Gemini API error details (key redacted): {st.session_state['gemini_error']}")

st.divider()
st.caption("Portfolio demo only. Model explanations are associations, not causal claims or guaranteed outcomes.")
