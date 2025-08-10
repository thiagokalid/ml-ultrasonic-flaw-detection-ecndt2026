# === Third-party libraries ===
import pandas as pd
import joblib
import time
import numpy as np

# --- PyOD ---
from pyod.models.iforest import IsolationForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.thresholds import CLUST
from pyod.models.inne import INNE

# Useful paths:
PKL_DATA_PATH = "../data/pkl/"

# --- Load the dataset ---
test_df, train_df = pd.read_pickle(PKL_DATA_PATH + 'test_df.pkl'),  pd.read_pickle(PKL_DATA_PATH + 'train_df.pkl')
X_test, X_train = joblib.load(PKL_DATA_PATH + "X_test.pkl"), joblib.load(PKL_DATA_PATH + "X_train.pkl")
y_test, y_train = joblib.load(PKL_DATA_PATH + "y_test.pkl"), joblib.load(PKL_DATA_PATH + "y_train.pkl")

print(f"Train: {len(train_df)} | Test: {len(test_df)} | "
      f"X_train: {X_train.shape} | X_test: {X_test.shape} | "
      f"y_train: {y_train.shape} | y_test: {y_test.shape} | "
      f"Test %: {len(test_df) / (len(train_df) + len(test_df)):.1%}")

# -- Training:
# models = ["knn", "iforest"]
models = ["lof"]
for model in models:
    match model:
        case "knn":
            clf = KNN(contamination=1/100, n_neighbors=15)
        case "iforest":
            clf = IsolationForest(contamination=1/100, n_estimators=100)
        case "ocsvm":
            clf = OCSVM(kernel='rbf', degree=3, nu=.5, contamination=1/100)
        case "inne":
            clf = INNE(contamination=1/100, n_estimators=300)
        case "lof":
            clf = LOF(n_neighbors=15, novelty=True, contamination=1/100)
        case "hbos":
            clf = HBOS(contamination=1/100, n_bins=10, alpha=0.1, tol=0.5)
        case "kmeans":
            clf = CLUST(method="kmeans")
        case _:
            print(f"Unknown model type: {model}")
            continue

    # Train the model:
    print("Training " + model + "...")
    t0 = time.time()
    clf.fit(X_train)
    elapsed_time = time.time() - t0
    print(f"Finished training in {elapsed_time:.2f} seconds.")

    # --- Predict anomalies ---
    print("Testing " + model + "...")
    t0 = time.time()
    y_pred = clf.predict(X_test)  # 0 = normal, 1 = anomaly (pyod uses this convention)
    tf = time.time() - t0
    print(f"Finished in {tf:.2f} seconds.")

    print("Computing anomaly scores " + model + "...")
    if hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)
    elif hasattr(clf, "decision_scores_"):
        y_scores = clf.decision_scores_
    else:
        print("Model has no decision_function or decision_scores_")
        y_scores = np.random.rand(len(y_test))


    tf = time.time() - t0
    print(f"Finished in {tf:.2f} seconds.")

    test_df[y_pred == 1].to_pickle(PKL_DATA_PATH + model + '_anomalies_df.pkl')
    joblib.dump(clf, PKL_DATA_PATH + model + '_clf.pkl')
    joblib.dump(y_pred, PKL_DATA_PATH + model + '_y_pred.pkl')
    joblib.dump(y_scores, PKL_DATA_PATH + model + '_y_scores.pkl')



