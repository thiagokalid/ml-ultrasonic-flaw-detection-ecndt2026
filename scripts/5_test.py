# === Standard library ===
import math

# === Third-party libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

import matplotlib
matplotlib.use("TkAgg")


# --- Scikit-learn ---
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    fbeta_score,
    precision_score,
    recall_score,
    roc_curve, auc
)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def create_annotations(cm_abs, cm_norm):
    """Create annotation strings with counts and percentages"""
    annotations = np.empty_like(cm_abs).astype(object)
    for i in range(cm_abs.shape[0]):
        for j in range(cm_abs.shape[1]):
            annotations[i, j] = f"{cm_abs[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
    return annotations



# Useful paths:
PKL_DATA_PATH = "../data/pkl/"

# Useful constants:
BETA_SCORE_CTE = 1

# Chosen model:
model = "lof"

# --- Load the dataset ---
test_df, train_td = pd.read_pickle(PKL_DATA_PATH + 'test_df.pkl'), pd.read_pickle(PKL_DATA_PATH + 'train_df.pkl')
X_test, X_train = joblib.load(PKL_DATA_PATH + "X_test.pkl"), joblib.load(PKL_DATA_PATH + "X_train.pkl")
y_test, y_train = joblib.load(PKL_DATA_PATH + "y_test.pkl"), joblib.load(PKL_DATA_PATH + "y_train.pkl")
y_scores = joblib.load(PKL_DATA_PATH + model + "_" + "y_scores.pkl")

#%% Load the model:
clf = joblib.load(PKL_DATA_PATH + model + "_clf.pkl")
# y_pred = clf.predict(X_test)
y_pred = joblib.load(PKL_DATA_PATH + model + "_" + "y_pred.pkl")

y_pred = np.where(y_pred > 0, 1, 0) # 0 normal and 1 anomaly
#%% Compute metrics
y_trivial = np.ones_like(y_pred, dtype=int)  # everything predicted as anomaly (1)

class_names = ['Normal', 'Anomaly']
# Raw counts
cm_model = confusion_matrix(y_test, y_pred, labels=[0,1])
cm_trivial = confusion_matrix(y_test, y_trivial, labels=[0,1])

# Normalize by true labels (rows)
cm_model_norm = cm_model / cm_model.sum(axis=1, keepdims=True)
cm_trivial_norm = cm_trivial / cm_trivial.sum(axis=1, keepdims=True)

# Handle division by zero if any row sums to zero
cm_model_norm = np.nan_to_num(cm_model_norm)
cm_trivial_norm = np.nan_to_num(cm_trivial_norm)
# -- Model metrics --
cm_model = confusion_matrix(y_test, y_pred, labels=[0, 1])
acc_model = accuracy_score(y_test, y_pred)
prec_model = precision_score(y_test, y_pred, zero_division=0)
rec_model = recall_score(y_test, y_pred, zero_division=0)
fbeta_model = fbeta_score(y_test, y_pred, beta=BETA_SCORE_CTE, zero_division=0)
bal_acc_model = balanced_accuracy_score(y_test, y_pred)

# -- Trivial metrics --
cm_trivial = confusion_matrix(y_test, y_trivial, labels=[0, 1])
acc_trivial = accuracy_score(y_test, y_trivial)
prec_trivial = precision_score(y_test, y_trivial, zero_division=0)
rec_trivial = recall_score(y_test, y_trivial, zero_division=0)
fbeta_trivial = fbeta_score(y_test, y_trivial, beta=BETA_SCORE_CTE, zero_division=0)
bal_acc_trivial = balanced_accuracy_score(y_test, y_trivial)

# Extract TN, FP, FN, TP from confusion matrices
tn_model, fp_model, fn_model, tp_model = cm_model.ravel()
tn_triv, fp_triv, fn_triv, tp_triv = cm_trivial.ravel()

# Compute Sensitivity (Recall)
sens_model = tp_model / (tp_model + fn_model) if (tp_model + fn_model) > 0 else 0
sens_triv = tp_triv / (tp_triv + fn_triv) if (tp_triv + fn_triv) > 0 else 0

# Compute Specificity
spec_model = tn_model / (tn_model + fp_model) if (tn_model + fp_model) > 0 else 0
spec_triv = tn_triv / (tn_triv + fp_triv) if (tn_triv + fp_triv) > 0 else 0

# Compute G-Mean
gmean_model = math.sqrt(sens_model * spec_model)
gmean_triv = math.sqrt(sens_triv * spec_triv)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Annotations for model
annot_model = create_annotations(cm_model, cm_model_norm)
sns.heatmap(
    cm_model_norm, annot=annot_model, fmt='', cmap='Blues', cbar=False,
    xticklabels=class_names, yticklabels=class_names, ax=axes[0]
)
axes[0].set_title("Model Prediction\nNormalized by True Label")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Annotations for trivial
annot_triv = create_annotations(cm_trivial, cm_trivial_norm)
sns.heatmap(
    cm_trivial_norm, annot=annot_triv, fmt='', cmap='Blues', cbar=False,
    xticklabels=class_names, yticklabels=class_names, ax=axes[1]
)
axes[1].set_title("Trivial Baseline\nNormalized by True Label")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


# Print metrics to terminal
print("\n=== Model Performance ===")
print(f"Accuracy        : {acc_model:.4f}")
print(f"Balanced Acc.   : {bal_acc_model:.4f}")
print(f"Precision       : {prec_model:.4f}")
print(f"Recall          : {rec_model:.4f}")
print(f"F{BETA_SCORE_CTE:.1f}-Score   : {fbeta_model:.4f}")
print(f"Sensitivity     : {sens_model:.4f}")
print(f"Specificity     : {spec_model:.4f}")
print(f"G-Mean          : {gmean_model:.4f}")

print("\n=== Trivial Baseline Performance ===")
print(f"Accuracy        : {acc_trivial:.4f}")
print(f"Balanced Acc.   : {bal_acc_trivial:.4f}")
print(f"Precision       : {prec_trivial:.4f}")
print(f"Recall          : {rec_trivial:.4f}")
print(f"F{BETA_SCORE_CTE:.1f}-Score   : {fbeta_trivial:.4f}")
print(f"Sensitivity     : {sens_triv:.4f}")
print(f"Specificity     : {spec_triv:.4f}")
print(f"G-Mean          : {gmean_triv:.4f}")


#%% Plot ROC curve:
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label="Random guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title(f"ROC Curve - {model.upper()}")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Precision–Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='green', lw=2, label=f"PR (AP = {ap_score:.2f})")
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve - {model.upper()}")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()

