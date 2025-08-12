import pandas as pd
import pickle
import os
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -----------------------
# Load processed data
# -----------------------
train_data = pd.read_csv(r"data\process\train_process_data.csv")
test_data = pd.read_csv(r"data\process\test_process_data.csv")

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

# Split features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# -----------------------
# Model 1: RandomForestClassifier
# -----------------------
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# -----------------------
# Model 2: VotingClassifier
# -----------------------
voting_clf = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("svc", SVC(probability=True, random_state=42))
    ],
    voting="soft"
)
voting_clf.fit(X_train, y_train)

# -----------------------
# Model 3: StackingClassifier
# -----------------------
stacking_clf = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42))
    ],
    final_estimator=LogisticRegression(),
    passthrough=False
)
stacking_clf.fit(X_train, y_train)

# -----------------------
# Save all models
# -----------------------
data_path = os.path.join("data", "models")
os.makedirs(data_path, exist_ok=True)

models = {
    "RandomForestClassifier_model.pkl": rf_clf,
    "VotingClassifier_model.pkl": voting_clf,
    "StackingClassifier_model.pkl": stacking_clf
}

for file_name, model in models.items():
    model_file = os.path.join(data_path, file_name)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {file_name} → {model_file}")
