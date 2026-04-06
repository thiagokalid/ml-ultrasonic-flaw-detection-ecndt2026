import warnings
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"multiprocessing\.queues"
)
warnings.filterwarnings(
    "ignore",
    message=r"y should not be presented in unsupervised learning",
    category=UserWarning,
    module=r"pyod\.models\.base"
)
warnings.filterwarnings(
    "ignore",
    message=r"The cost of rejection must be positive. It has been set to the contamination rate.",
    category=UserWarning,
    module=r"pyod\.models\.base"
)

# === Third-party libraries ===
import pandas as pd
import joblib
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
import copy
from pathlib import Path
import json

# --- PyOD ---
from pyod.models.cblof import CBLOF
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
from pyod.models.kpca import KPCA


CV_GRIDSEARCH = False
PREDICT_WITH_REJECTION = True
DEFAULT_CONTAMINATION = 6/100
# SCORING = make_scorer(fbeta_score, beta=2)
SCORING = "f1"

COMMON_PARAMS_GRID = {
    "contamination": [2/100, 5/100, 7/100]
}

MODELS_PARAMS_GRID = {
    "lof":{
        "n_neighbors": [10, 15, 20, 25],
        "p": [1],
        "novelty": [True],
        "n_jobs": [-1]
    },
    "iforest":{
        "n_estimators": [25, 50, 100, 200],
        "n_jobs": [-1],
        "behaviour": ["new"],
        "random_state": [42],
    },
    "ocsvm":{
        "kernel":['linear'],
    },
    "hbos":{
        "n_bins": [2, 5, 10, 15],
        "alpha": [.05, .1, .2, 1, 2]
    },
    "inne":{
        "n_estimators": [50, 100, 200]
    },
    "cblof":{
        "n_clusters": [5, 7, 10, 12, 15],
        "alpha": [0.6, 0.7, 0.8, 0.9, 0.95]
    },
    # "kpca":{
    #     "kernel": ["rbf", "linear"],
    #     "alpha": [.1, 1, 2],
    #     "n_components": [20, 50, 100]
    # }
}

# Useful paths:
DATASET_PATH = Path("../data/dataset")
MODELS_PATH = Path("../data/models")

# --- Load the dataset ---
test_df, validation_df, train_df = pd.read_pickle(DATASET_PATH / 'test_df.pkl'),  pd.read_pickle(DATASET_PATH / 'validation_df.pkl'),  pd.read_pickle(DATASET_PATH / 'train_df.pkl')
X_test, X_validation, X_train = joblib.load(DATASET_PATH / "X_test.pkl"), joblib.load(DATASET_PATH / "X_validation.pkl"), joblib.load(DATASET_PATH / "X_train.pkl")
y_test, y_validation, y_train = joblib.load(DATASET_PATH / "y_test.pkl"), joblib.load(DATASET_PATH / "y_validation.pkl"), joblib.load(DATASET_PATH / "y_train.pkl")

validation_df['y_pred'] = 0

elapsed_time_seconds = {
    "train": 0.,
    "test": 0.,
    "number_of_test_samples": len(X_test),
    "number_of_train_samples": len(X_train),
}

#%% Train different models:
for ith_model, model in enumerate(MODELS_PARAMS_GRID.keys()):
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
            clf = LOF(contamination=DEFAULT_CONTAMINATION, p=1, n_neighbors=15, n_jobs=-1)
        case "cblof":
            clf = CBLOF(contamination=DEFAULT_CONTAMINATION, n_jobs=-1)
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
        case "kpca":
            clf = KPCA(contamination=DEFAULT_CONTAMINATION)
        case _:
            print(f"Unknown model type: {model}")
            continue

    curr_elapsed_time = copy.deepcopy(elapsed_time_seconds)

    print("Training " + model + f"... (Model {ith_model + 1} out of {len(MODELS_PARAMS_GRID)})")
    t0 = time.time()
    if CV_GRIDSEARCH:
        search_space = MODELS_PARAMS_GRID[model] | COMMON_PARAMS_GRID
        cv = GridSearchCV(clf, param_grid=search_space, scoring=SCORING, n_jobs=-1)
        cv.fit(X_validation, y_validation)
        clf = cv.best_estimator_


    clf.fit(X_train, y_train)

    elapsed_time = time.time() - t0
    curr_elapsed_time["train"] = elapsed_time
    print(f"Finished training in {elapsed_time:.2f} seconds.")


    # --- Predict anomalies ---
    print("Testing " + model + "...")
    if PREDICT_WITH_REJECTION:
        t0 = time.time()
        y_pred = clf.predict_with_rejection(X_test, T=32, delta=.05)
        y_pred[y_pred == -2] = 0
        elapsed_time = time.time() - t0
        curr_elapsed_time["test"] = elapsed_time
        print(f"Finished in {elapsed_time:.2f} seconds.")

        y_pred_val = clf.predict_with_rejection(X_validation, T=32, delta=.05)
        y_pred_val[y_pred_val == -2] = 0

        y_pred_train = clf.predict_with_rejection(X_train, T=32, delta=.05)
        y_pred_train[y_pred_train == -2] = 0
    else:
        y_pred = clf.predict(X_test)

        y_pred_val = clf.predict(X_validation)

        y_pred_test = clf.predict(X_test)

    print("Computing anomaly scores " + model + "...")
    if hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)

        y_scores_val = clf.decision_function(X_validation)

        y_scores_train = clf.decision_function(X_train)
    # elif hasattr(clf, "decis2*exp(-32)ion_scores_"):
    #     y_scores = clf.decision_scores_
    #
    #     y_scores_val = clf.decision_scores_
    # else:
    #     print("Model has no decision_function or decision_scores_")
    #     y_scores = np.random.rand(len(y_test))
    #
    #     y_scores_val = np.random.rand(len(y_test))


    tf = time.time() - t0
    print(f"Finished in {tf:.2f} seconds.")

    (MODELS_PATH / model).mkdir(parents=True, exist_ok=True)

    prediction_df = test_df.copy()
    prediction_df["y_pred"] = y_pred
    prediction_df.to_pickle(MODELS_PATH / model / 'prediction_df.pkl')

    joblib.dump(clf, MODELS_PATH / model / 'clf.pkl')
    joblib.dump(y_pred, MODELS_PATH / model / 'y_pred.pkl')
    joblib.dump(y_scores, MODELS_PATH / model / 'y_scores.pkl')


    prediction_val_df = validation_df.copy()
    prediction_val_df["y_pred"] = y_pred_val

    prediction_train_df = train_df.copy()
    prediction_train_df["y_pred"] = y_pred_train


    predictions_df_show = pd.concat([prediction_df, prediction_val_df, prediction_train_df], axis=0, ignore_index=True)
    y_scores_show = np.concatenate((y_scores, y_scores_val, y_scores_train), axis=0)
    y_pred_show = np.concatenate((y_pred, y_pred_val, y_pred_train), axis=0)

    predictions_df_show.to_pickle(MODELS_PATH / model / 'prediction_df_show.pkl')
    joblib.dump(y_pred_show, MODELS_PATH / model / 'y_pred_show.pkl')
    joblib.dump(y_scores_show, MODELS_PATH / model / 'y_scores_show.pkl')
    with open("elapsed_time.json", "w") as f:
        json.dump(curr_elapsed_time, f, indent=4)



