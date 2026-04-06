# === Third-party libraries ===
import pandas as pd
import joblib
import numpy as np
import time
from pathlib import Path


# Useful paths:
DATA_PATH = Path("../data")
DATASET_PATH = DATA_PATH / "dataset"
MODELS_PATH = DATA_PATH / "models"
model = 'threshold'



# --- Load the dataset ---
dataset = pd.read_pickle(DATASET_PATH / 'dataset.pkl')
test_df, validation_df, train_df = pd.read_pickle(DATASET_PATH / 'test_df.pkl'),  pd.read_pickle(DATASET_PATH / 'validation_df.pkl'),  pd.read_pickle(DATASET_PATH / 'train_df.pkl')
X_test, X_validation, X_train = joblib.load(DATASET_PATH / "X_test.pkl"), joblib.load(DATASET_PATH / "X_validation.pkl"), joblib.load(DATASET_PATH / "X_train.pkl")
y_test, y_validation, y_train = joblib.load(DATASET_PATH / "y_test.pkl"), joblib.load(DATASET_PATH / "y_validation.pkl"), joblib.load(DATASET_PATH / "y_train.pkl")

elapsed_time_seconds = {
    "train": 0.,
    "test": 0.,
    "number_of_test_samples": len(X_test),
    "number_of_train_samples": len(X_train),
}

validation_df['y_pred'] = 0

t0 = time.time()
for file in dataset["filename"].unique():

    file_mask = dataset["filename"] == file
    shots = dataset.loc[file_mask, "shot"].unique()

    for shot in shots:

        mask = file_mask & (dataset["shot"] == shot)

        max_pixel = dataset.loc[mask, "tiles"].apply(np.max).max()

        dataset.loc[mask, "max_tiles"] = max_pixel


threshold_test = dataset.loc[test_df.index, "max_tiles"].to_numpy()
threshold_val = dataset.loc[validation_df.index, "max_tiles"].to_numpy()
threshold_train = dataset.loc[train_df.index, "max_tiles"].to_numpy()

y_pred_test = np.array([
    np.count_nonzero(x > t) >= 0.01 * len(x)
    for x, t in zip(test_df["tiles"].to_numpy(), threshold_test)
])
elapsed_time_seconds["test"] = time.time() - t0

y_pred_val = np.array([
    np.count_nonzero(x > t) >= 0.01 * len(x)
    for x, t in zip(validation_df["tiles"].to_numpy(), threshold_val)
])

y_pred_train = np.array([
    np.count_nonzero(x > t) >= 0.01 * len(x)
    for x, t in zip(train_df["tiles"].to_numpy(), threshold_train)
])


y_pred = y_pred_test

prediction_df = test_df.copy()
prediction_df["y_pred"] = y_pred

# Ensure path exists:
(MODELS_PATH / model).mkdir(parents=True, exist_ok=True)

prediction_df.to_pickle(MODELS_PATH / model / 'prediction_df.pkl')

joblib.dump(y_pred, MODELS_PATH / model / 'y_pred.pkl')
joblib.dump(y_pred, MODELS_PATH / model / 'y_scores.pkl')


prediction_val_df = validation_df.copy()
prediction_val_df["y_pred"] = y_pred_val

prediction_test_df = test_df.copy()
prediction_test_df["y_pred"] = y_pred_test


predictions_df_show = pd.concat([prediction_df, prediction_val_df, prediction_test_df], axis=0, ignore_index=True)
predictions_df_show.to_pickle(MODELS_PATH / model / 'prediction_df_show.pkl')