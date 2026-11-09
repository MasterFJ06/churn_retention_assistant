"""Train and evaluate a first XGBoost churn model using the Gold table."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "telco_gold.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


def main() -> None:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold table not found at {GOLD_PATH}. Run src/prepare_data.py first."
        )

    gold = pd.read_csv(GOLD_PATH)
    X = gold.drop(columns="Churn")
    y = gold["Churn"]

    # Keep categorical columns readable in Gold; encode them only after splitting.
    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numbers", "passthrough", numeric_features),
        ]
    )

    # Stratification preserves the same churn/stay ratio in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # SQLite keeps experiment history and model metadata in one local database.
    tracking_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("churn-retention-assistant")
    run = mlflow.start_run(run_name="xgboost-baseline")
    mlflow.log_params(
        {
            "model": "XGBClassifier",
            "test_size": 0.20,
            "random_state": 42,
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "risk_threshold": 0.50,
            "categorical_feature_count": len(categorical_features),
            "numeric_feature_count": len(numeric_features),
        }
    )
    pipeline.fit(X_train, y_train)

    # The model's probability is a score; 0.50 is just our initial decision cutoff.
    churn_probability = pipeline.predict_proba(X_test)[:, 1]
    predictions = (churn_probability >= 0.50).astype("int64")

    metrics = {
        "roc_auc": roc_auc_score(y_test, churn_probability),
        "f1": f1_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    mlflow.log_metrics(metrics)
    mlflow.log_dict(
        {
            "labels": ["stayed", "churned"],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
        "evaluation/confusion_matrix.json",
    )

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Metrics on the held-out test set (initial threshold = 0.50):")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.3f}")
    print(f"  F1:       {metrics['f1']:.3f}")
    print(f"  Precision:{metrics['precision']:.3f}")
    print(f"  Recall:   {metrics['recall']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print("Confusion matrix (rows=actual [stay, churn], columns=predicted [stay, churn]):")
    print(matrix)

    examples = X_test.head(5).copy()
    examples["actual_churn"] = y_test.loc[examples.index]
    examples["predicted_churn_probability"] = pipeline.predict_proba(examples.drop(columns="actual_churn"))[:, 1]
    print("\nExample held-out customer risk scores:")
    print(examples[["tenure", "Contract", "MonthlyCharges", "actual_churn", "predicted_churn_probability"]].to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    mlflow.sklearn.log_model(
        pipeline,
        name="churn-pipeline",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )
    print(f"\nSaved fitted pipeline to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Logged metrics and model to MLflow run: {run.info.run_id}")
    mlflow.end_run()


if __name__ == "__main__":
    main()
