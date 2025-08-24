from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
import joblib

data = load_iris(as_frame=True)
df = data.frame.copy()
df["target"] = df["target"].map(dict(enumerate(data.target_names)))
df.to_csv("iris.csv", index=False)

# random_forest_classification.py
# pip install pandas scikit-learn matplotlib joblib

# --------------------
# 1) CONFIG — EDIT THESE
# --------------------
CSV_PATH = "iris.csv"          # <- path to your CSV
TARGET_COL = "target"               # <- name of your target column
ID_COLS_TO_DROP = []                # <- e.g. ["id", "user_id"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------
# 2) LOAD DATA
# --------------------
df = pd.read_csv(CSV_PATH)

# Optionally drop ID / leakage columns
cols_to_drop = [c for c in ID_COLS_TO_DROP if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Split features/target
if TARGET_COL not in df.columns:
    raise ValueError(f"TARGET_COL '{TARGET_COL}' not found in columns: {list(df.columns)}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# --------------------
# 3) AUTO-DETECT COLUMN TYPES
# --------------------
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

if not cat_cols and not num_cols:
    raise ValueError("No usable feature columns detected. Check your data types.")

# --------------------
# 4) PREPROCESSING & MODEL PIPELINE
# --------------------
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # Random Forest doesn’t need scaling
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    class_weight="balanced",   # helpful if classes are imbalanced
    random_state=RANDOM_STATE
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", rf)
])

# --------------------
# 5) TRAIN/TEST SPLIT & TRAIN
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

pipe.fit(X_train, y_train)

# --------------------
# 6) EVALUATION
# --------------------
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n=== Metrics ===")
print(f"Accuracy:     {acc:.4f}")
print(f"F1 (macro):   {f1_macro:.4f}\n")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ROC-AUC (binary or multiclass)
try:
    if hasattr(pipe.named_steps["rf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)
        classes = pipe.named_steps["rf"].classes_
        if len(classes) == 2:
            rocauc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            rocauc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        print(f"ROC-AUC:      {rocauc:.4f}\n")
except Exception as e:
    print(f"(Skipped ROC-AUC: {e})\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=pipe.named_steps["rf"].classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.named_steps["rf"].classes_)
fig = plt.figure()
disp.plot(values_format="d")
plt.title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.show()

# --------------------
# 7) TOP FEATURE IMPORTANCES
# --------------------
try:
    # Get feature names post-preprocessing (works in sklearn >= 1.0)
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["rf"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)

    top_n = 20
    print(f"\n=== Top {top_n} Features ===")
    print(fi.head(top_n).to_string(index=False))

    # Optional: quick bar plot
    top = fi.head(top_n).sort_values("importance")
    plt.figure()
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances — Random Forest")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"(Skipped feature importance breakdown: {e})")

# --------------------
# 8) SAVE THE MODEL
# --------------------
MODEL_PATH = "random_forest_model.joblib"
joblib.dump(pipe, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
