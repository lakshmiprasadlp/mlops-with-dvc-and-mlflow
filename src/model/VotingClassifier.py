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
from logger import get_logger  # ✅ custom logger

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

logger = get_logger("VotingClassifier")


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
    logger.info("Initializing DagsHub and MLflow...")
    dagshub.init(repo_owner='lakshmiprasadlp',
                 repo_name='MLOPS_with_DVC_and_Mlflow',
                 mlflow=True)
    mlflow.set_experiment("Model_Comparison")

    cfg_path = r"yaml\VotingClassifier.yaml"
    logger.info(f"Loading config from {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    model_dir = "data/models"
    metrics_dir = "data/metrics"
    plots_dir = os.path.join(metrics_dir, "confusion_matrices")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Output directories ready.")

    logger.info("Loading processed datasets...")
    train_data = pd.read_csv("data/process/train_process_data.csv")
    test_data = pd.read_csv("data/process/test_process_data.csv")
    logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    with mlflow.start_run(run_name="VotingClassifier"):
        logger.info("Logging config to MLflow...")
        mlflow.log_artifact(cfg_path, artifact_path="config")
        mlflow.log_params(_flatten(config))

        logger.info("Creating base RandomForestClassifier estimators...")
        rf1 = RandomForestClassifier(**config["estimators"]["rf"])
        rf2 = RandomForestClassifier(**config["estimators"]["rf2"])
        rf3 = RandomForestClassifier(**config["estimators"]["rf3"])

        logger.info("Building VotingClassifier model...")
        model = VotingClassifier(
            estimators=[("rf1", rf1), ("rf2", rf2), ("rf3", rf3)],
            voting=config["voting"]
        )

        logger.info("Training VotingClassifier...")
        model.fit(X_train, y_train)
        logger.info("Training completed.")

        model_path = os.path.join(model_dir, "VotingClassifier_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved at {model_path}")

        logger.info("Logging model to MLflow/DagsHub...")
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Generating predictions and classification report...")
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
        logger.info("Metrics logged to MLflow.")

        metrics_json_path = os.path.join(metrics_dir, "VotingClassifier_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(metrics_json_path, artifact_path="metrics")
        logger.info(f"Metrics JSON saved at {metrics_json_path}")

        cm = confusion_matrix(y_test, y_pred)
        cm_img = os.path.join(plots_dir, "VotingClassifier_cm.png")
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - VotingClassifier")
        plt.tight_layout()
        plt.savefig(cm_img)
        plt.close()
        mlflow.log_artifact(cm_img, artifact_path="artifacts")
        logger.info(f"Confusion matrix PNG saved at {cm_img}")

        cm_csv = os.path.join(plots_dir, "VotingClassifier_cm.csv")
        pd.DataFrame(cm).to_csv(cm_csv, index=False)
        mlflow.log_artifact(cm_csv, artifact_path="artifacts")
        logger.info(f"Confusion matrix CSV saved at {cm_csv}")

    logger.info("VotingClassifier run completed successfully.")


if __name__ == "__main__":
    main()
