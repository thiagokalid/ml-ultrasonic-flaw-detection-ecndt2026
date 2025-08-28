# === Third-party libraries ===
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.lof import LOF

# Useful paths:
PKL_DATA_PATH = "../data/pkl/"

# --- Load the dataset ---
test_df, train_df = pd.read_pickle(PKL_DATA_PATH + 'lof_prediction_df.pkl'),  pd.read_pickle(PKL_DATA_PATH + 'train_df.pkl')
X_test, X_train = joblib.load(PKL_DATA_PATH + "X_test.pkl"), joblib.load(PKL_DATA_PATH + "X_train.pkl")
y_test, y_train = joblib.load(PKL_DATA_PATH + "y_test.pkl"), joblib.load(PKL_DATA_PATH + "y_train.pkl")


subroi_idx = 100

mask1 = np.logical_and(X_train[f'subroi_{subroi_idx}'] == True, test_df['filename'] == "2nd_row_v1.m2k")

mask2 = np.logical_and(X_test[f'subroi_{subroi_idx}'] == True, train_df['filename'] == "2nd_row_v1.m2k")

X_train_selected, y_train_selected = X_train[mask1], y_train[mask1]
X_test_selected, y_test_selected = X_test[mask2], y_test[mask2]

x1_train, x2_train = X_train_selected['fft_abs_mean'].values.reshape(-1, 1), X_train_selected['fft_abs_median'].values.reshape(-1, 1)
x1_test, x2_test = X_test_selected['fft_abs_mean'].values.reshape(-1, 1), X_test_selected['fft_abs_median'].values.reshape(-1, 1)

# Combine features
X_train_feat = np.hstack([x1_train, x2_train])
X_test_feat  = np.hstack([x1_test, x2_test])

# --- Fit LOF on training data ---
lof = LOF(n_neighbors=20, contamination=1/100, novelty=True)  # novelty=True allows predicting on new samples
lof.fit(X_train_feat)

# --- Predict anomalies on train and test ---
y_test_pred  = lof.predict(X_test_feat) # 0 = inlier, 1 = outlier

# --- Compute global limits for plotting ---
x_all = np.concatenate([x1_train.ravel(), x1_test.ravel()])
y_all = np.concatenate([x2_train.ravel(), x2_test.ravel()])
xlim = (x_all.min(), x_all.max())
ylim = (y_all.min(), y_all.max())

# --- Datasets for plotting ---
datasets = [
    ("Train (GT)", X_train_feat, y_train_selected),  # Ground truth train
    ("Test (GT)",  X_test_feat,  y_test_selected),   # Ground truth test
    ("LOF Prediction", np.vstack([X_test_feat]),
     np.hstack([y_test_pred]))         # LOF predictions combined
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, (title, X_feat, y_labels) in zip(axes, datasets):
    # Plot inliers
    inliers = y_labels == 0
    ax.plot(X_feat[inliers, 0], X_feat[inliers, 1], 'o', color='black', label='Inlier')

    # Plot outliers
    outliers = y_labels == 1
    ax.plot(X_feat[outliers, 0], X_feat[outliers, 1], 'o', color='red', markersize=6, label='Outlier')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std")
    ax.legend()

plt.tight_layout()
plt.show()


