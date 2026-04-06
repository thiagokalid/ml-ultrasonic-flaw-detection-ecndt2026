# === Standard library ===
import math

# === Third-party libraries ===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, fbeta_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve
)
from datetime import datetime

from pathlib import Path
from scripts.utils import make_confusion_matrix

# --- Useful paths & constants ---
DATA_PATH = Path("../data/")
DATASET_PATH = DATA_PATH / "dataset"
MODELS_PATH = DATA_PATH / "models"
PLOT_CONFUSION_MATRIX = True
BETA_SCORE_CTE = 2

# Load dataset once
X_test = joblib.load(DATASET_PATH / "X_test.pkl")
y_test = joblib.load(DATASET_PATH / "y_test.pkl")

# Candidate models
models = ["lof"]

# Store results
results = []
roc_curves = {}
for model in models:
    # Load predictions and scores
    y_pred = joblib.load(MODELS_PATH / model / f"y_pred.pkl")
    y_scores = joblib.load(MODELS_PATH / model / f"y_scores.pkl")

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

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_curves[model] = (fpr, tpr, roc_auc)

    # Append row
    results.append({
        "Model": model,
        "Accuracy": acc,
        # "Balanced Acc.": bal_acc,
        "Precision": prec,
        "Recall": rec,
        f"F{BETA_SCORE_CTE}-Score": fbeta,
        # "Sensitivity": sens,
        # "Specificity": spec,
        # "G-Mean": gmean,
        "ROC AUC": roc_auc,
        # "AP Score": ap_score
    })

    if PLOT_CONFUSION_MATRIX:
        make_confusion_matrix(cm, group_names=["Normal", "Anomaly"], categories=["Normal", "Anomaly"], cbar=False, figsize=(7 * .55, 6 * .55))
        plt.tight_layout()
        plt.savefig(f"../figures/{model}_confusion_matrix.pdf", dpi=300)

    # Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.round(3).to_latex(f"../figures/{model}_results.tex")

pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)       # don't wrap to next line
pd.set_option("display.max_colwidth", None)  # show full column content
df_str = results_df.round(3).to_string(index=False)
print(df_str)

# Create timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results_{timestamp}.txt"

# Save to txt
with open(filename, "w") as f:
    f.write(df_str)

print(f"Saved as {filename}")


#%% === Plot ROC AUC curves for all models ===
plt.figure(figsize=(7 * .55, 6 * .55))
i = 0
colors = ["k", "#FF1F5B", "#009ADE", "#AF58BA", "#FFC61E"]
for model, (fpr, tpr, roc_auc) in roc_curves.items():
    if model == "threshold":
        continue
    if model == "lof":
        alpha = 1
    else:
        alpha = .5
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f"{model} (AUC = {roc_auc:.3f})", alpha=alpha)
    i += 1

plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Characteristic Operational do Receptor (ROC) Curves")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/roc_curves.pdf", dpi=300)
plt.show()