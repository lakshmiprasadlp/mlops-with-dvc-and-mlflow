import os
import pickle
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    # Load config
    with open(r"src\yaml\VotingClassifier.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Paths
    model_dir = "data/models"
    metrics_dir = "data/metrics"
    plots_dir = os.path.join(metrics_dir, "confusion_matrices")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    train_data = pd.read_csv("data/process/train_process_data.csv")
    test_data = pd.read_csv("data/process/test_process_data.csv")
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    # Create base models
    rf1 = RandomForestClassifier(**config["estimators"]["rf"])
    rf2 = RandomForestClassifier(**config["estimators"]["rf2"])
    rf3 = RandomForestClassifier(**config["estimators"]["rf3"])

    # Create Voting Classifier
    model = VotingClassifier(
        estimators=[("rf1", rf1), ("rf2", rf2), ("rf3", rf3)],
        voting=config["voting"]
    )
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(model_dir, "VotingClassifier_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Save metrics
    with open(os.path.join(metrics_dir, "VotingClassifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - VotingClassifier")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "VotingClassifier_cm.png"))
    plt.close()

if __name__ == "__main__":
    main()
