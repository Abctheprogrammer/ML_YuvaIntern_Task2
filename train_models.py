"""Week 2: Supervised Machine Learning Models
Runs regression and classification experiments and saves metrics, predictions, and charts.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
RANDOM_STATE = 42


def run_regression():
    dataset = load_diabetes(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    rows, predictions = [], pd.DataFrame({"actual": y_test.reset_index(drop=True)})
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
        })
        predictions[name] = pred
    results = pd.DataFrame(rows).sort_values("RMSE")
    results.to_csv(OUT / "regression_metrics.csv", index=False)
    predictions.to_csv(DATA / "regression_predictions.csv", index=False)

    ax = results.set_index("model")[["MAE", "RMSE"]].plot(kind="bar", figsize=(9, 5))
    ax.set_title("Regression Model Comparison")
    ax.set_ylabel("Error (lower is better)")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "regression_comparison.png", dpi=160)
    plt.close()
    return results


def run_classification():
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
        })
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=dataset.target_names
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("-", "")
        plt.savefig(OUT / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close()
    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    results.to_csv(OUT / "classification_metrics.csv", index=False)

    ax = results.set_index("model")[["accuracy", "precision", "recall", "f1_score"]].plot(
        kind="bar", figsize=(10, 5), ylim=(0, 1.05)
    )
    ax.set_title("Classification Model Comparison")
    ax.set_ylabel("Score (higher is better)")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "classification_comparison.png", dpi=160)
    plt.close()
    return results


def main():
    print("\\n=== Regression Results (Diabetes dataset) ===")
    reg = run_regression()
    print(reg.round(4).to_string(index=False))
    print("\\n=== Classification Results (Breast Cancer Wisconsin dataset) ===")
    clf = run_classification()
    print(clf.round(4).to_string(index=False))
    print(f"\\nDone. Metrics and charts saved under: {OUT}")
    print("Educational benchmark only; not intended for medical diagnosis.")


if __name__ == "__main__":
    main()
