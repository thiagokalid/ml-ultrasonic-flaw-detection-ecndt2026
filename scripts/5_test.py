# === Standard library ===
import math

# === Third-party libraries ===
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, fbeta_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# --- Useful paths & constants ---
PKL_DATA_PATH = "../data/pkl/"
BETA_SCORE_CTE = 1

# Load dataset once
X_test = joblib.load(PKL_DATA_PATH + "X_test.pkl")
y_test = joblib.load(PKL_DATA_PATH + "y_test.pkl")

# Candidate models
models = ["lof", "hbos", "iforest"]

# Store results
results = []

for model in models:
    # Load predictions and scores
    y_pred = joblib.load(PKL_DATA_PATH + f"{model}_y_pred.pkl")
    y_scores = joblib.load(PKL_DATA_PATH + f"{model}_y_scores.pkl")

    # Convert prediction to {0,1}
    y_pred = np.where(y_pred > 0, 1, 0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    fbeta = fbeta_score(y_test, y_pred, beta=BETA_SCORE_CTE, zero_division=0)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = math.sqrt(sens * spec)

    roc_auc = roc_auc_score(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)

    # Append row
    results.append({
        "Model": model,
        "Accuracy": acc,
        "Balanced Acc.": bal_acc,
        "Precision": prec,
        "Recall": rec,
        f"F{BETA_SCORE_CTE}-Score": fbeta,
        "Sensitivity": sens,
        "Specificity": spec,
        "G-Mean": gmean,
        "ROC AUC": roc_auc,
        "AP Score": ap_score
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)       # don't wrap to next line
pd.set_option("display.max_colwidth", None)  # show full column content
print(results_df.round(4))
