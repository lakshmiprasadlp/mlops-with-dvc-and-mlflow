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
# Model 1: RandomForestClassifier (with custom hyperparameters)
# -----------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=15,           # limit depth
    min_samples_split=4,    # min samples to split node
    min_samples_leaf=2,     # min samples per leaf
    bootstrap=True,         # use bootstrap sampling
    random_state=42
)
rf_clf.fit(X_train, y_train)

# -----------------------
# Model 2: VotingClassifier (with tuned base models)
# -----------------------
gb_clf = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)

svc_clf = SVC(
    C=1.5,
    kernel="rbf",
    gamma="scale",
    probability=True,
    random_state=42
)

voting_clf = VotingClassifier(
    estimators=[
        ("rf", rf_clf),
        ("gb", gb_clf),
        ("svc", svc_clf)
    ],
    voting="soft"
)
voting_clf.fit(X_train, y_train)

# -----------------------
# Model 3: StackingClassifier (with tuned base models)
# -----------------------
stacking_clf = StackingClassifier(
    estimators=[
        ("rf", rf_clf),
        ("gb", gb_clf)
    ],
    final_estimator=LogisticRegression(
        solver="liblinear",
        C=0.8,
        max_iter=500
    ),
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
