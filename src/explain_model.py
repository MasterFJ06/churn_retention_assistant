"""Explain one held-out XGBoost prediction with SHAP."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "telco_gold.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
PLOT_PATH = PROJECT_ROOT / "reports" / "shap_waterfall.png"


def main() -> None:
    if not GOLD_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Gold data or fitted model is missing. Run prepare_data.py and "
            "train_model.py first."
        )

    gold = pd.read_csv(GOLD_PATH)
    X = gold.drop(columns="Churn")
    y = gold["Churn"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = joblib.load(MODEL_PATH)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    true_positive_positions = np.flatnonzero(
        (probabilities >= 0.50) & (y_test.to_numpy() == 1)
    )
    if len(true_positive_positions):
        row_position = true_positive_positions[np.argmax(probabilities[true_positive_positions])]
    else:
        row_position = int(np.argmax(probabilities))

    customer = X_test.iloc[[row_position]]
    actual_churn = int(y_test.iloc[row_position])
    probability = float(pipeline.predict_proba(customer)[0, 1])

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    encoded = preprocessor.transform(customer)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    raw_shap_values = explainer.shap_values(encoded, check_additivity=False)
    if isinstance(raw_shap_values, list):
        raw_shap_values = raw_shap_values[-1]
    raw_shap_values = np.asarray(raw_shap_values)
    if raw_shap_values.ndim == 3:
        values = raw_shap_values[0, :, -1]
    elif raw_shap_values.ndim == 2:
        values = raw_shap_values[0]
    else:
        values = raw_shap_values.reshape(-1)

    base_value = np.asarray(explainer.expected_value)
    if base_value.ndim:
        base_value = base_value.reshape(-1)[-1]
    else:
        base_value = base_value.item()

    print("Held-out customer example")
    print(f"  Tenure: {customer.iloc[0]['tenure']} months")
    print(f"  Contract: {customer.iloc[0]['Contract']}")
    print(f"  Monthly charges: ${customer.iloc[0]['MonthlyCharges']:.2f}")
    print(f"  Actual outcome: {'churned' if actual_churn else 'stayed'}")
    print(f"  Model churn score: {probability:.1%}")
    print("\nLargest SHAP factors (positive moves the model score toward churn):")

    order = np.argsort(np.abs(values))[::-1]
    for index in order[:8]:
        direction = "toward churn" if values[index] > 0 else "away from churn"
        print(f"  {feature_names[index]}: {values[index]:+.3f} ({direction})")

    explanation = shap.Explanation(
        values=values,
        base_values=float(base_value),
        data=encoded[0],
        feature_names=feature_names,
    )
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved local SHAP waterfall plot to: {PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print("SHAP values here are in the model's raw-score (log-odds) space, not percentage points.")


if __name__ == "__main__":
    main()
