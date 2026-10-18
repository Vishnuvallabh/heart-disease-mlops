"""Tune two models with 5-fold CV, log everything to MLflow, save the best."""
import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline

from features import FEATURES, build_preprocessor

CLEAN_PATH = "data/processed/heart_clean.csv"


def get_models():
    """Each model with the settings GridSearchCV should try."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [None, 5, 10]},
        ),
    }


def evaluate(model, X_test, y_test):
    """Test-set metrics for a fitted model."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return pred, prob, {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, prob),
    }


def save_plots(name, y_test, pred, prob):
    """Confusion matrix and ROC curve, saved and logged to MLflow."""
    os.makedirs("plots", exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(f"Confusion Matrix - {name}")
    cm_path = f"plots/{name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title(f"ROC Curve - {name}")
    roc_path = f"plots/{name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path)


def main():
    df = pd.read_csv(CLEAN_PATH)
    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_experiment("heart-disease")

    best_name, best_model, best_metrics = None, None, None

    for name, (model, grid) in get_models().items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("preprocess", build_preprocessor()),
                             ("model", model)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc")
            search.fit(X_train, y_train)

            # one child run per grid combination
            res = search.cv_results_
            for i, params in enumerate(res["params"]):
                with mlflow.start_run(run_name=f"{name}_trial_{i}",
                                      nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_roc_auc",
                                      res["mean_test_score"][i])

            best = search.best_estimator_
            pred, prob, test_metrics = evaluate(best, X_test, y_test)
            metrics = {"cv_roc_auc": search.best_score_, **test_metrics}

            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            save_plots(name, y_test, pred, prob)
            mlflow.sklearn.log_model(best, name="model",
                                     serialization_format="cloudpickle")

            print(f"{name}: " + ", ".join(
                f"{k}={v:.3f}" for k, v in metrics.items()))

            if (best_metrics is None
                    or metrics["cv_roc_auc"] > best_metrics["cv_roc_auc"]):
                best_name, best_model, best_metrics = name, best, metrics

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")

    metadata = {
        "model_name": best_name,
        "selected_by": "cv_roc_auc",
        "features": FEATURES,
        "metrics": {k: round(v, 4) for k, v in best_metrics.items()},
        "sklearn_version": sklearn.__version__,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFinal model: {best_name} -> models/model.joblib")


if __name__ == "__main__":
    main()
