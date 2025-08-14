# === Third-party libraries ===
import pandas as pd
import joblib
import time
import numpy as np
from sklearn.model_selection import GridSearchCV

# --- PyOD ---
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.thresholds import CLUST
from pyod.models.inne import INNE
from pyod.models.abod import ABOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.pca import PCA
from pyod.models.xgbod import XGBOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.anogan import AnoGAN

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
            clf = IForest(contamination=1/100, n_estimators=200, max_features=1, behaviour='new')
        case "ocsvm":
            clf = OCSVM(kernel='rbf', degree=3, nu=.5, contamination=1/100)
        case "inne":
            clf = INNE(contamination=1/100, n_estimators=100)
        case "lof":
            clf = LOF(n_neighbors=15, novelty=True, contamination=1/100, p=1)
            param_grid = {
                'contamination': [1/100, .1/100],
                'n_neighbors': [10, 15, 30, 45],
                "p": [1, 2]
            }
        case "hbos":
            clf = HBOS(contamination=1/100, n_bins=10, alpha=0.1, tol=0.5)
        case "kmeans":
            clf = CLUST(method="kmeans")
        case "abod":
            clf = ABOD(contamination=1/100)
        case "deepsvdd":
            clf = DeepSVDD(contamination=1/100)
        case "pca":
            clf = PCA(n_components=25, contamination=1/100)
        case "xgbod":
            init_clf = []
            clf = XGBOD(n_jobs=4,learning_rate=0.01 )
        case "autoencoder":
            clf = AutoEncoder(contamination=1/100)
        case "anogan":
            clf = AnoGAN(contamination=1/100, verbose=1, preprocessing=False)
        case _:
            print(f"Unknown model type: {model}")
            continue

    # Hyperparameter tuning:
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')

    # Fit with hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Choose best estimator:
    clf = grid_search.best_estimator_

    # Train the model:
    print("Training " + model + "...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed_time = time.time() - t0
    print(f"Finished training in {elapsed_time:.2f} seconds.")

    # --- Predict anomalies ---
    print("Testing " + model + "...")
    t0 = time.time()
    # y_pred = clf.predict_with_rejection(X_test, T=36, return_stats=False, delta=0.1, c_fp=1, c_fn=1, c_r=-1) # 0 = normal, 1 = anomaly (pyod uses this convention)
    y_pred = clf.predict(X_test)
    # print("Number of rejections: ", np.count_nonzero(y_pred == -2))
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



