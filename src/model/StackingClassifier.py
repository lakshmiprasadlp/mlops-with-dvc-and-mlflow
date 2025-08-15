import os
import pickle
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score

def _flatten(d, parent_key=""):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        else:
            items[new_key] = v
    return items

def main():
    dagshub.init(repo_owner='lakshmiprasadlp', repo_name='MLOPS_with_DVC_and_Mlflow', mlflow=True)
    mlflow.set_experiment("Model_Comparison")

    cfg_path = r"yaml\StackingClassifier.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    model_dir = "data/models"
    metrics_dir = "data/metrics"
    plots_dir = os.path.join(metrics_dir, "confusion_matrices")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    train_data = pd.read_csv("data/process/train_process_data.csv")
    test_data  = pd.read_csv("data/process/test_process_data.csv")
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test,  y_test  = test_data.iloc[:, :-1].values,  test_data.iloc[:, -1].values

    with mlflow.start_run(run_name="StackingClassifier"):
        mlflow.log_artifact(cfg_path, artifact_path="config")
        mlflow.log_params(_flatten(config))

        rf1 = RandomForestClassifier(**config["estimators"]["rf"])
        rf2 = RandomForestClassifier(**config["estimators"]["rf2"])
        rf3 = RandomForestClassifier(**config["estimators"]["rf3"])

        model = StackingClassifier(
            estimators=[("rf1", rf1), ("rf2", rf2), ("rf3", rf3)],
            final_estimator=LogisticRegression(**config["final_estimator"])
        )
        model.fit(X_train, y_train)

        model_path = os.path.join(model_dir, "StackingClassifier_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, artifact_path="model")

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        unique_labels = list(pd.Series(y_test).unique())
        recalls = recall_score(y_test, y_pred, labels=unique_labels, average=None, zero_division=0)
        lab2rec = {str(l): float(r) for l, r in zip(unique_labels, recalls)}

        mlflow.log_metrics({
            "accuracy": float(report.get("accuracy", 0.0)),
            "recall_class_1": lab2rec.get("1", 0.0),
            "recall_class_0": lab2rec.get("0", 0.0),
            "f1_score_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        })

        metrics_json_path = os.path.join(metrics_dir, "StackingClassifier_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(metrics_json_path, artifact_path="metrics")

        cm = confusion_matrix(y_test, y_pred)
        cm_img = os.path.join(plots_dir, "StackingClassifier_cm.png")
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - StackingClassifier")
        plt.tight_layout()
        plt.savefig(cm_img)
        plt.close()
        mlflow.log_artifact(cm_img, artifact_path="artifacts")

        cm_csv = os.path.join(plots_dir, "StackingClassifier_cm.csv")
        pd.DataFrame(cm).to_csv(cm_csv, index=False)
        mlflow.log_artifact(cm_csv, artifact_path="artifacts")

if __name__ == "__main__":
    main()
