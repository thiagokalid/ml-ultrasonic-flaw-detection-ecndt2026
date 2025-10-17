import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('TkAgg')

# Data root:
PKL_DATA_PATH = "../data/pkl/"

#
selected_filename = "TuboNovo_260shots.m2k"
selected_features = [
    ("mean", "std")
]

df = pd.read_pickle(PKL_DATA_PATH + "test_df.pkl")
X = joblib.load(PKL_DATA_PATH + "X_test.pkl")
y_true = joblib.load(PKL_DATA_PATH + "y_test.pkl")
y_lof = joblib.load(PKL_DATA_PATH + "models/" + "lof_y_pred.pkl")


#%%
print(np.sum(df.loc[df["filename"] == selected_filename, "contain_flaw"]))
mask = (df["filename"] == selected_filename) & (df["subroi_idx"] == 33)
X = X[mask]
y_true = y_true[mask]
y_lof = y_lof[mask]


feature1 = "mean"
feature2 = "std"

colours = ["k", "r"]
markersize=[3, 9]
x1, x2 = X[feature1], X[feature2]

plt.figure()
for i in range(len(y_true)):
    label = int(y_true[i])
    plt.plot(x1.iloc[i], x2.iloc[i], 'o', color=colours[label], markersize=markersize[label])
plt.show()