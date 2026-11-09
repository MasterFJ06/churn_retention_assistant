"""Create cleaned Silver and model-ready Gold tables from the raw IBM Telco CSV."""

from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "telco_silver.csv"
GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "telco_gold.csv"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Download it into data/raw first."
        )

    # Bronze: read the original file without changing it.
    bronze = pd.read_csv(RAW_PATH)
    print(f"Bronze rows and columns: {bronze.shape}")

    # Silver: normalize text and convert TotalCharges from text to a number.
    silver = bronze.copy()
    silver.columns = silver.columns.str.strip()
    text_columns = silver.select_dtypes(include=["object", "str"]).columns
    for column in text_columns:
        silver[column] = silver[column].str.strip()

    silver["TotalCharges"] = pd.to_numeric(
        silver["TotalCharges"].replace("", pd.NA), errors="coerce"
    )

    missing_total_charges = silver["TotalCharges"].isna()
    unexpected_missing = missing_total_charges & silver["tenure"].ne(0)
    if unexpected_missing.any():
        raise ValueError(
            "Found missing TotalCharges for a customer with non-zero tenure; "
            "inspect these records before filling values."
        )

    # The blank TotalCharges values belong to brand-new customers (tenure = 0).
    # Zero is therefore the sensible value for their total billed amount.
    silver.loc[missing_total_charges, "TotalCharges"] = 0.0
    silver["SeniorCitizen"] = silver["SeniorCitizen"].astype("int64")
    silver["tenure"] = silver["tenure"].astype("int64")

    churn_values = set(silver["Churn"].dropna().unique())
    if churn_values != {"Yes", "No"}:
        raise ValueError(f"Unexpected values in Churn column: {churn_values}")

    SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(SILVER_PATH, index=False)

    # Gold: remove the identifier and encode the label as 1 (churn) / 0 (stay).
    gold = silver.drop(columns=["customerID"]).copy()
    gold["Churn"] = gold["Churn"].map({"No": 0, "Yes": 1}).astype("int64")
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(GOLD_PATH, index=False)

    print(f"Silver saved: {SILVER_PATH.relative_to(PROJECT_ROOT)} ({silver.shape})")
    print(f"Gold saved:   {GOLD_PATH.relative_to(PROJECT_ROOT)} ({gold.shape})")
    print(f"Missing TotalCharges filled for zero-tenure customers: {int(missing_total_charges.sum())}")
    print(f"Gold target counts (0=stayed, 1=churned): {gold['Churn'].value_counts().sort_index().to_dict()}")

    # SQL analytics on Gold using DuckDB; the model pipeline will handle encoding.
    connection = duckdb.connect()
    connection.register("gold", gold)
    churn_by_contract = connection.execute(
        """
        SELECT
            Contract,
            COUNT(*) AS customers,
            ROUND(100.0 * AVG(Churn), 2) AS churn_rate_pct
        FROM gold
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
        """
    ).df()
    print("\nChurn rate by contract (DuckDB SQL):")
    print(churn_by_contract.to_string(index=False))
    connection.close()


if __name__ == "__main__":
    main()
