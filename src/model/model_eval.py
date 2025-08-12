import os
import pickle
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, 
                           precision_score, 
                           recall_score, 
                           f1_score, 
                           confusion_matrix)

def load_test_data():
    """Load and prepare test data"""
    test_data = pd.read_csv("data/process/test_process_data.csv")
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics"""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

def save_confusion_matrix(y_test, y_pred, model_name, save_dir):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{model_name}_confusion_matrix.png")
    plt.close()

def main():
    # Setup paths
    metrics_dir = "data/metrics"
    models_dir = "data/models"
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Initialize results storage
    all_metrics = {}
    
    # Evaluate each model
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pkl"):
            model_name = model_file.replace("_model.pkl", "")
            model_path = os.path.join(models_dir, model_file)
            
            # Load model
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            all_metrics[model_name] = metrics
            
            # Save visualization
            y_pred = model.predict(X_test)
            save_confusion_matrix(y_test, y_pred, model_name, metrics_dir)
    
    # Save metrics
    os.makedirs(metrics_dir, exist_ok=True)
    with open(f"{metrics_dir}/performance.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("Evaluation completed successfully!")
    print(f"Metrics saved to: {metrics_dir}/performance.json")

if __name__ == "__main__":
    main()