import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, 
                           precision_score, 
                           recall_score, 
                           f1_score, 
                           confusion_matrix)

def main():
    # Setup paths
    metrics_dir = "data/metrics"
    plots_dir = os.path.join(metrics_dir, "confusion_matrices")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load test data
    test_data = pd.read_csv("data/process/test_process_data.csv")
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    # Initialize results
    results = {}
    
    # Evaluate each model
    for model_file in os.listdir("data/models"):
        if model_file.endswith(".pkl"):
            model_name = model_file.replace("_model.pkl", "")
            model_path = os.path.join("data/models", model_file)
            
            # Load and evaluate model
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_test)
            results[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            
            # Save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{model_name}_cm.png"))
            plt.close()
    
    # Save metrics
    with open(os.path.join(metrics_dir, "performance.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()