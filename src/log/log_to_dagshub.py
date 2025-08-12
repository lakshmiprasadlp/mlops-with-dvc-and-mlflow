import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------
# DagsHub + MLflow setup
# -----------------------
dagshub.init(repo_owner='lakshmiprasadlp', repo_name='MLOPS_with_DVC_and_Mlflow', mlflow=True)
mlflow.set_experiment("Model Evaluation")

# -----------------------
# Load processed test data
# -----------------------
test_data_path = r"data\process\test_process_data.csv"
train_data_path = r"data\process\train_process_data.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# -----------------------
# Model files
# -----------------------
model_files = {
    "RandomForestClassifier": r"data\models\RandomForestClassifier_model.pkl",
    "VotingClassifier": r"data\models\VotingClassifier_model.pkl",
    "StackingClassifier": r"data\models\StackingClassifier_model.pkl"
}

# -----------------------
# Output directory
# -----------------------
metrics_path = os.path.join("data", "metrics")
os.makedirs(metrics_path, exist_ok=True)

# -----------------------
# Start MLflow Run
# -----------------------
with mlflow.start_run(run_name="Evaluate_Models"):
    all_metrics = {}

    # Log dataset to MLflow/DagsHub
    mlflow.log_artifact(train_data_path, artifact_path="dataset")
    mlflow.log_artifact(test_data_path, artifact_path="dataset")

    for model_name, model_file in model_files.items():
        model = pickle.load(open(model_file, "rb"))
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        all_metrics[model_name] = {
            'accuracy': acc,
            'precision': pre,
            'recall': recall,
            'f1_score': f1score
        }

        # Log metrics
        mlflow.log_metric(f"{model_name}_accuracy", acc)
        mlflow.log_metric(f"{model_name}_precision", pre)
        mlflow.log_metric(f"{model_name}_recall", recall)
        mlflow.log_metric(f"{model_name}_f1_score", f1score)

        # Confusion Matrix Image
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"]
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_file = os.path.join(metrics_path, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_file)
        plt.close()

        # Log confusion matrix image
        mlflow.log_artifact(cm_file, artifact_path="confusion_matrices")

    # Save metrics as CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_csv_file = os.path.join(metrics_path, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_file)

    # Log metrics CSV
    mlflow.log_artifact(metrics_csv_file, artifact_path="metrics")

print("✅ Logged dataset, metrics, and images to DagsHub/MLflow")
