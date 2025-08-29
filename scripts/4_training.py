import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Third-party libraries ===
import pandas as pd
import joblib
import time
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
import json
from tqdm import tqdm


# --- PyOD ---
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.thresholds import CLUST
from pyod.models.inne import INNE
from pyod.models.abod import ABOD
from pyod.models.pca import PCA
from pyod.models.xgbod import XGBOD



CV_GRIDSEARCH = True
DEFAULT_CONTAMINATION = 1/100
# SCORING = make_scorer(fbeta_score, beta=3)
SCORING = "f1"

COMMON_PARAMS_GRID = {
    "contamination": [.1/100, .5/100, 1/100, 1.5/100, 2/100, 3/100]
}

MODELS_PARAMS_GRID = {
    "lof":{
        "n_neighbors": [2, 5, 10, 15, 20, 25, 30],
        "p": [1, 2],
        "novelty": [True],
        "n_jobs": [-1]
    },
    "iforest":{
        "n_estimators": [25, 50, 100, 200],
        "n_jobs": [-1],
        "behaviour": ["new"],
        "random_state": [42],
    },
    # "pca":{
    #     "n_components": [5, 10, 25, 50, 100],
    # },
    # "hbos":{
    #     "n_bins": [2, 5, 10, 15],
    #     "alpha": [.05, .1, .2, 1, 2]
    # },
    # "inne":{
    #     "n_estimators": [50, 100, 200]
    # }
}

# Useful paths:
PKL_DATA_PATH = "../data/pkl/"
MODEL_PATH = PKL_DATA_PATH + "models/"

# --- Load the dataset ---
test_df, validation_df, train_df = pd.read_pickle(PKL_DATA_PATH + 'test_df.pkl'),  pd.read_pickle(PKL_DATA_PATH + 'validation_df.pkl'),  pd.read_pickle(PKL_DATA_PATH + 'train_df.pkl')
X_test, X_validation, X_train = joblib.load(PKL_DATA_PATH + "X_test.pkl"), joblib.load(PKL_DATA_PATH + "X_validation.pkl"), joblib.load(PKL_DATA_PATH + "X_train.pkl")
y_test, y_validation, y_train = joblib.load(PKL_DATA_PATH + "y_test.pkl"), joblib.load(PKL_DATA_PATH + "y_validation.pkl"), joblib.load(PKL_DATA_PATH + "y_train.pkl")

#%% Train different models:
for model in tqdm(MODELS_PARAMS_GRID.keys()):
    match model:
        case "knn":
            clf = KNN(contamination=DEFAULT_CONTAMINATION, n_neighbors=15)
        case "iforest":
            clf = IForest(contamination=DEFAULT_CONTAMINATION, n_estimators=100, random_state=42)
        case "ocsvm":
            clf = OCSVM(contamination=DEFAULT_CONTAMINATION)
        case "inne":
            clf = INNE(contamination=DEFAULT_CONTAMINATION, n_estimators=100)
        case "lof":
            clf = LOF(contamination=DEFAULT_CONTAMINATION, p=2, n_neighbors=15, n_jobs=-1)
        case "hbos":
            clf = HBOS(contamination=DEFAULT_CONTAMINATION, n_bins=10)
        case "kmeans":
            clf = CLUST(contamination=DEFAULT_CONTAMINATION, method="kmeans")
        case "abod":
            clf = ABOD(contamination=DEFAULT_CONTAMINATION)
        case "pca":
            clf = PCA()
        case "xgbod":
            init_clf = []
            clf = XGBOD(n_jobs=4,learning_rate=0.01 )
        case _:
            print(f"Unknown model type: {model}")
            continue

    print("Training " + model + "...")
    t0 = time.time()
    if not CV_GRIDSEARCH:
        clf.fit(X_train, y_train)
    else:
        search_space = MODELS_PARAMS_GRID[model] | COMMON_PARAMS_GRID
        cv = GridSearchCV(clf, param_grid=search_space, scoring=SCORING, n_jobs=-1)
        cv.fit(X_validation, y_validation)
        clf = cv.best_estimator_
        clf.fit(X_train, y_train)

    elapsed_time = time.time() - t0
    print(f"Finished training in {elapsed_time:.2f} seconds.")


    # --- Predict anomalies ---
    print("Testing " + model + "...")
    t0 = time.time()
    y_pred = clf.predict(X_test)
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

    prediction_df = test_df.copy()
    prediction_df["y_pred"] = y_pred
    prediction_df.to_pickle(MODEL_PATH + model + '_prediction_df.pkl')
    joblib.dump(clf, MODEL_PATH + model + '_clf.pkl')
    joblib.dump(y_pred, MODEL_PATH + model + '_y_pred.pkl')
    joblib.dump(y_scores, MODEL_PATH + model + '_y_scores.pkl')



