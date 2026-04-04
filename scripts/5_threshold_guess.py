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

# === Third-party libraries ===
import pandas as pd
import joblib
import numpy as np

CV_GRIDSEARCH = False
PREDICT_WITH_REJECTION = True
DEFAULT_CONTAMINATION = 5/100
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
}

# Useful paths:
PKL_DATA_PATH = "../data/pkl/"
MODEL_PATH = PKL_DATA_PATH + "models/"
model = 'threshold'

# --- Load the dataset ---
dataset = pd.read_pickle(PKL_DATA_PATH + 'dataset.pkl')
test_df, validation_df, train_df = pd.read_pickle(PKL_DATA_PATH + 'test_df.pkl'),  pd.read_pickle(PKL_DATA_PATH + 'validation_df.pkl'),  pd.read_pickle(PKL_DATA_PATH + 'train_df.pkl')
X_test, X_validation, X_train = joblib.load(PKL_DATA_PATH + "X_test.pkl"), joblib.load(PKL_DATA_PATH + "X_validation.pkl"), joblib.load(PKL_DATA_PATH + "X_train.pkl")
y_test, y_validation, y_train = joblib.load(PKL_DATA_PATH + "y_test.pkl"), joblib.load(PKL_DATA_PATH + "y_validation.pkl"), joblib.load(PKL_DATA_PATH + "y_train.pkl")

validation_df['y_pred'] = 0

for file in dataset["filename"].unique():

    file_mask = dataset["filename"] == file
    shots = dataset.loc[file_mask, "shot"].unique()

    for shot in shots:

        mask = file_mask & (dataset["shot"] == shot)

        max_pixel = dataset.loc[mask, "sub_sscan"].apply(np.max).max()

        dataset.loc[mask, "max_subsscan"] = max_pixel





threshold_test = dataset.loc[test_df.index, "max_subsscan"].to_numpy()
threshold_val = dataset.loc[validation_df.index, "max_subsscan"].to_numpy()
threshold_train = dataset.loc[train_df.index, "max_subsscan"].to_numpy()

y_pred_test = np.array([
    np.count_nonzero(x > t) >= 0.01 * len(x)
    for x, t in zip(test_df["sub_sscan"].to_numpy(), threshold_test)
])

y_pred_val = np.array([
    np.count_nonzero(x > t) >= 0.01 * len(x)
    for x, t in zip(validation_df["sub_sscan"].to_numpy(), threshold_val)
])

y_pred_train = np.array([
    np.count_nonzero(x > t) >= 0.01 * len(x)
    for x, t in zip(train_df["sub_sscan"].to_numpy(), threshold_train)
])


y_pred = y_pred_test

prediction_df = test_df.copy()
prediction_df["y_pred"] = y_pred
prediction_df.to_pickle(MODEL_PATH + model + '_prediction_df.pkl')

joblib.dump(y_pred, MODEL_PATH + model + '_y_pred.pkl')
joblib.dump(y_pred, MODEL_PATH + model + '_y_scores.pkl')


prediction_val_df = validation_df.copy()
prediction_val_df["y_pred"] = y_pred_val

prediction_test_df = test_df.copy()
prediction_test_df["y_pred"] = y_pred_test


predictions_df_show = pd.concat([prediction_df, prediction_val_df, prediction_test_df], axis=0, ignore_index=True)
predictions_df_show.to_pickle(MODEL_PATH + model + '_prediction_df_show.pkl')