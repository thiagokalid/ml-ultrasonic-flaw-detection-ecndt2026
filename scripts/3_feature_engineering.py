# === Standard library ===
import math

# === Third-party libraries ===
import numpy as np
import pandas as pd
import pywt
import joblib

# --- Scikit-learn ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Data root:
PKL_DATA_PATH = "../data/pkl/"

# --- Parameters ---
N_PCA_COMPONENTS = 20  # e.g., reduce angle dimension to 5
BETA_SCORE_CTE = 1

# -- Wavelet feature extractor from a 2D array --
def extract_wavelet_features_2d(image, wavelet='sym4', level=2):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    features = []

    def compute_features(subband):
        flat = subband.ravel()
        # energy = np.sum(flat ** 2)
        # mean_val = np.mean(flat)
        # std_val = np.std(flat)
        # power = np.abs(flat) ** 2
        # power_sum = np.sum(power)
        # ent = entropy(power / power_sum) if power_sum != 0 else 0

        energy = np.sum(flat ** 2)
        l1_norm = np.max(np.sum(np.abs(subband), axis=0))
        f_norm = np.mean(np.abs(subband)**2)**(1/2)
        delta_norm = np.max(np.abs(subband))

        return [l1_norm, f_norm]

    # LL (approximation)
    features.extend(compute_features(coeffs[0]))

    # Detail subbands at each level: (LH, HL, HH)
    for (cH, cV, cD) in coeffs[1:]:
        features.extend(compute_features(cH))
        features.extend(compute_features(cV))
        features.extend(compute_features(cD))

    return np.array(features)


# --- Function to pad arrays to target shape ---
def pad_to_shape(arr, target_shape):
    padded = np.zeros(target_shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    padded[slices] = arr[slices]
    return padded


# --- Load the dataset ---
df = pd.read_pickle(PKL_DATA_PATH + 'dataset_annotated.pkl')

# # Define training mask correctly:
# training_mask = (
#     (
#         (df['filename'].apply(lambda x: x[-6:] != "v2.m2k")) &
#         (df['contain_flaw'] == False)
#         # (df['ith_shot'] >= 40)
#     )
#     |
#     (
#         (df['filename'].apply(lambda x: x[:12] == "passive_dir_")) &
#         (df['contain_flaw'] == False) &
#         (df['filename'].apply(lambda x: x[-6:] != "v2.m2k"))
#         # (df['ith_shot'] <= 5 | (df['ith_shot'] >= 20))
#     )
#     |
#     (df['filename'].apply(lambda x: x[3:8] == "_row_")) &
#     (df['contain_flaw'] == False) &
#     (df['filename'].apply(lambda x: x[-6:] != "v2.m2k"))
# )
#
# # Normalize S-scan:
# df['sub_sscan'] = df['sub_sscan'] / df['sscan_max']
#
# # Extract sets:
# train_df = df[training_mask]
# test_df = df[~training_mask]

print(len(df))
# df['sub_sscan'] = df['sub_sscan'] / df['sscan_max']

TRAINING_METHOD = "Semi-supervised"
if TRAINING_METHOD == "Semi-supervised":
    # Separate flaws and non-flaws
    flaws_df = df[df['contain_flaw'] == True]
    non_flaws_df = df[df['contain_flaw'] == False]

    # Desired test size including all flaws
    target_test_ratio = 0.30

    # Number of samples for test excluding flaws
    total_count = len(df)
    num_flaws = len(flaws_df)
    num_non_flaws_test = max(0, int(target_test_ratio * total_count) - num_flaws)

    # Split non-flaws into train/test
    non_flaws_train, non_flaws_test = train_test_split(
        non_flaws_df,
        test_size=num_non_flaws_test,
        random_state=42,
        shuffle=True
    )

    test_df = pd.concat([flaws_df, non_flaws_test]).reset_index(drop=True)
    train_df = non_flaws_train.reset_index(drop=True)

    # print(len(train_df))
    # print(len(test_df))
    # print(len(train_df) + len(test_df))
    #
    # # Sanity check
    # print(f"Train proportion: {len(train_df) / total_count:.2%}")
    # print(f"Test proportion:  {len(test_df) / total_count:.2%}")
    # print(f"Flaws in train?   {train_df['contain_flaw'].any()}")  # Should be False
    # print(f"Flaws in test?    {test_df['contain_flaw'].any()}")  # Should be True
elif TRAINING_METHOD == "Supervised":
    train_df, test_df = train_test_split(df, test_size=.3)
else:
    raise ValueError("Invalid TRAINING_METHOD")

# Combine flaws with test portion of non-flaws



#%%
# -- PCA features --

# --- Find max shape on training set ---
max_shape = tuple(np.max([arr.shape for arr in train_df['sub_sscan']], axis=0))

# --- Pad and flatten ---
X_pxs_train = np.stack(train_df['sub_sscan'].apply(lambda x: pad_to_shape(x, max_shape).ravel()))
X_pxs_test = np.stack(test_df['sub_sscan'].apply(lambda x: pad_to_shape(x, max_shape).ravel()))

# X_pxs_train = np.log10(X_pxs_train + 1E-6)
# X_pxs_test = np.log10(X_pxs_test + 1E-6)

# --- Fit PCA on training ---
pca = PCA(n_components=N_PCA_COMPONENTS)
pca.fit(X_pxs_train)

# --- Transform both sets ---
X_pxs_train_pca = pca.transform(X_pxs_train)
X_pxs_test_pca = pca.transform(X_pxs_test)

#%%
# -- FFT features --
# print("FFT before")

X_fft_train = train_df['sub_sscan'].apply(np.fft.fft2)
X_fft_test = test_df['sub_sscan'].apply(np.fft.fft2)

# # --- Find max shape on training set ---
# max_shape = tuple(np.max([arr.shape for arr in X_fft_train], axis=0))
#
# # --- Pad and flatten ---
# X_fft_train = np.stack(X_fft_train.apply(lambda x: pad_to_shape(x, max_shape).ravel()))
# X_fft_test = np.stack(X_fft_test.apply(lambda x: pad_to_shape(x, max_shape).ravel()))
#
# # --- Fit PCA on training ---
# pca_real = PCA(n_components=10)
# pca_real.fit(np.real(X_fft_train))
#
# # --- Transform both sets ---
# X_fft_train_pca_real = pca_real.transform(np.real(X_fft_train))
# X_fft_test_pca_real = pca_real.transform(np.real(X_fft_test))
#
# # --- Fit PCA on training ---
# pca_imag = PCA(n_components=10)
# pca_imag.fit(np.imag(X_fft_train))
#
# # --- Transform both sets ---
# X_fft_train_pca_imag = pca_imag.transform(np.imag(X_fft_train))
# X_fft_test_pca_imag = pca_imag.transform(np.imag(X_fft_test))


#%% Select features for training and testing:

X_train_parts, X_test_parts = [], []

for df, parts, pca_features, X_fft in zip(
        [train_df, test_df],
        [X_train_parts, X_test_parts],
        [X_pxs_train_pca, X_pxs_test_pca],
        [X_fft_train, X_fft_test]
):
    # -- One-hot encode subroi_idx --
    subroi_ohe = pd.get_dummies(df['subroi_idx'], prefix='subroi')
    parts.append(subroi_ohe)

    # -- Statistical features --
    for operation in [np.mean, np.std, np.max, np.min, np.ptp, np.median]:
        operation_name = operation.__name__
        operation_db = lambda x: operation(x)
        parts.append(df['sub_sscan'].apply(operation_db).to_frame(name=operation_name))

    # -- Wavelet features --
    # wavelet_features = np.stack(df['sub_sscan'].apply(extract_wavelet_features_2d))
    # wavelet_df = pd.DataFrame(wavelet_features, index=df.index)
    # wavelet_df.columns = [f"wavelet_{i}" for i in range(wavelet_df.shape[1])]
    # parts.append(wavelet_df)

    # -- PCA features --
    pca_df = pd.DataFrame(pca_features, index=df.index)
    pca_df.columns = [f"pca_{i}" for i in range(pca_df.shape[1])]
    parts.append(pca_df)

    # -- FFT features --
    # pca_df = pd.DataFrame(pca_real, index=df.index)
    # pca_df.columns = [f"pca_real_{i}" for i in range(pca_df.shape[1])]
    # parts.append(pca_df)
    #
    # pca_df = pd.DataFrame(pca_imag, index=df.index)
    # pca_df.columns = [f"pca_imag_{i}" for i in range(pca_df.shape[1])]
    # parts.append(pca_df)

    # -- FFT Features --
    for operation in [np.mean, np.median, np.argmax, np.argmin]:
        operation_name = operation.__name__
        operation_abs = lambda x: operation(np.abs(x))
        parts.append(X_fft.apply(operation_abs).to_frame(name="fft_abs_" + operation_name))

        # operation_phase = lambda x: operation(np.angle(x))
        # parts.append(X_fft.apply(operation_phase).to_frame(name="fft_phase_" + operation_name))


# --- Concatenate features ---
X_train = pd.concat(X_train_parts, axis=1)
X_test = pd.concat(X_test_parts, axis=1)

# Keep feature names before scaling
feature_names = X_train.columns

# Align columns in test to match training
X_test = X_test.reindex(columns=feature_names, fill_value=0)

# --- Standardize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Back to DataFrames
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# Labels
y_train = train_df['contain_flaw']
y_test = test_df['contain_flaw']

# Split into validation and test
test_df, validation_df, X_test, X_validation, X_test_df, X_validation_df, y_test, y_validation = train_test_split(
    test_df, X_test, X_test_df, y_test, test_size=0.3, random_state=42
)


# Add split flag
X_train_df["split"] = "train"
X_test_df["split"] = "test"
X_validation_df["split"] = "validation"

# Combine
X_combined = pd.concat([X_train_df, X_test_df, X_validation_df], ignore_index=True)
y_combined = pd.concat([y_train, y_test, y_validation], ignore_index=True)

# # -- Select the most meaningful features:
# selector = SelectKBest(f_classif, k=len(X_train.columns) - 5)
# selector.fit(X_train, y_train)
#
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)

# %% Print dataset info:
n_samples = len(y_combined)

# Sanity checks with pretty formatting
print(f"{'Train proportion:':25} {len(y_train) / total_count:.2%}")
print(f"{'Validation proportion:':25} {len(y_validation) / total_count:.2%}")
print(f"{'Test proportion:':25} {len(y_test) / total_count:.2%}")

print(f"{'Flaws in train?':25} {train_df['contain_flaw'].any()}")
print(f"{'Flaws in validation?':25} {validation_df['contain_flaw'].any()} "
      f"(Contamination: {np.sum(validation_df['contain_flaw']) / len(y_validation):.2%})")
print(f"{'Flaws in test?':25} {test_df['contain_flaw'].any()} "
      f"(Contamination: {np.sum(test_df['contain_flaw']) / len(y_test):.2%})")


# %% Save X and y to disk
joblib.dump(X_train, PKL_DATA_PATH + "X_train.pkl")
joblib.dump(X_test, PKL_DATA_PATH + "X_test.pkl")
joblib.dump(X_validation, PKL_DATA_PATH + "X_validation.pkl")
joblib.dump(y_train.to_numpy(), PKL_DATA_PATH + "y_train.pkl")
joblib.dump(y_test.to_numpy(), PKL_DATA_PATH + "y_test.pkl")
joblib.dump(y_validation.to_numpy(), PKL_DATA_PATH + "y_validation.pkl")
test_df.to_pickle(PKL_DATA_PATH + "test_df.pkl")
train_df.to_pickle(PKL_DATA_PATH + "train_df.pkl")
validation_df.to_pickle(PKL_DATA_PATH + "validation_df.pkl")
X_combined.to_pickle(PKL_DATA_PATH + "X_combined.pkl")
y_test.to_pickle(PKL_DATA_PATH + "y_combined.pkl")
