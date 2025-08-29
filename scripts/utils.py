from framework.post_proc import envelope as envelope_fun
from framework import file_m2k
import numpy as np

def get_reduction_method(method: str):
    match method:
        case "mean":
            return np.mean
        case "median":
            return np.median
        case _:
            raise ValueError(f"Unknown reduction method {method}")

def apply_scale(sscan: np.ndarray, scale: str) -> np.ndarray:
    match scale:
        case "linear":
            return sscan
        case "log":
            LOG_CTE = 1e-6
            return np.log(sscan + LOG_CTE)
        case _:
            raise ValueError(f"Unknown scale {scale}")

def apply_preprocessing(sscan, envelope: bool, normalize: bool) -> np.ndarray:
    new_sscan = np.copy(sscan)

    if envelope:
        new_sscan = envelope_fun(new_sscan, axis=0)

    if normalize:
        new_sscan = (new_sscan - new_sscan.min()) / (new_sscan.max() - new_sscan.min())

    return new_sscan

def get_ref(channels_shape, ref_filepath, compute_subtraction, m2k_configs) -> np.ndarray:
    if compute_subtraction:
        data_ref = file_m2k.read(ref_filepath, *m2k_configs)
        if channels_shape == data_ref.ascan_data[..., 0].shape:
            return data_ref.ascan_data[..., 0]
        else:
            raise ValueError
    else:
        return np.zeros(channels_shape)

def get_annotation_insp(annotations_insp, filename, shot):
    XXXXX = "active_dir_xl_focused_1degree_v1"  # example pattern

    matches = [d for d in annotations_insp["images"] if f"{XXXXX}_m2k" in d["file_name"]]

    for m in matches:
        print(m)


def split_sscan2subrois(df_rows, sscan: np.ndarray, shot, time_grid: np.ndarray, alpha_grid: np.ndarray, t_outer, t_inner, params) -> None:
    NUM_SUBROIS_XAXIS = params["num_subrois_xaxis"]
    NUM_SUBROIS_ZAXIS = params["num_subrois_zaxis"]

    # Splitting the S-scan into smallers S-scans:
    subroi_xaxis_borders = np.linspace(alpha_grid.min(), alpha_grid.max(), NUM_SUBROIS_XAXIS)
    subroi_yaxis_borders = np.linspace(t_outer, t_inner, NUM_SUBROIS_ZAXIS)

    sscan_max = sscan.max()

    for i in range(1, len(subroi_xaxis_borders)):
        for j in range(1, len(subroi_yaxis_borders)):
            x_start, x_end = subroi_xaxis_borders[i - 1], subroi_xaxis_borders[i]
            y_start, y_end = subroi_yaxis_borders[j - 1], subroi_yaxis_borders[j]

            sscan_limits = [(x_start, x_end), (y_start, y_end)]

            # Safe indexing
            t_start_idx = np.searchsorted(time_grid, y_start)
            t_end_idx = np.searchsorted(time_grid, y_end)
            a_start_idx = np.searchsorted(alpha_grid, x_start)
            a_end_idx = np.searchsorted(alpha_grid, x_end)

            sub_sscan = sscan[t_start_idx:t_end_idx, a_start_idx:a_end_idx]
            subroi_idx = (i - 1) * (len(subroi_yaxis_borders) - 1) + (j - 1)



            # Append to dict:
            df_rows["delta_t"].append(time_grid[1] - time_grid[0])
            df_rows["delta_ang"].append(alpha_grid[1] - alpha_grid[0])
            df_rows["shot"].append(shot)
            df_rows["subroi_idx"].append(subroi_idx)
            df_rows["t_outer"].append(t_outer)
            df_rows["t_inner"].append(t_inner)
            df_rows["sscan_max"].append(sscan_max)
            df_rows["contain_flaw"].append(np.nan)
            df_rows["subroi_limits"].append(sscan_limits)
            df_rows["sub_sscan"].append(sub_sscan)
    return None

def mark_subsscans(df, mask_resized, time_grid, alpha_grid, t_outer, t_inner, params, thresh=5/100):
    NUM_SUBROIS_XAXIS = params["num_subrois_xaxis"]
    NUM_SUBROIS_ZAXIS = params["num_subrois_zaxis"]

    # Splitting the S-scan into smallers S-scans:
    subroi_xaxis_borders = np.linspace(alpha_grid.min(), alpha_grid.max(), NUM_SUBROIS_XAXIS)
    subroi_yaxis_borders = np.linspace(t_outer, t_inner, NUM_SUBROIS_ZAXIS)

    flaw_size = np.sum(mask_resized)

    number_of_subrois = (params["num_subrois_xaxis"] - 1) * (params["num_subrois_zaxis"] - 1)

    has_flaw = np.zeros(number_of_subrois)

    for i in range(1, len(subroi_xaxis_borders)):
        for j in range(1, len(subroi_yaxis_borders)):
            x_start, x_end = subroi_xaxis_borders[i - 1], subroi_xaxis_borders[i]
            y_start, y_end = subroi_yaxis_borders[j - 1], subroi_yaxis_borders[j]

            # Safe indexing
            t_start_idx = np.searchsorted(time_grid, y_start)
            t_end_idx = np.searchsorted(time_grid, y_end)
            a_start_idx = np.searchsorted(alpha_grid, x_start)
            a_end_idx = np.searchsorted(alpha_grid, x_end)

            cropped_mask = mask_resized[t_start_idx:t_end_idx, a_start_idx:a_end_idx]
            # if np.sum(cropped_mask) > (flaw_size * thresh):
            #     import matplotlib.pyplot as plt
            #     plt.figure()
            #     plt.imshow(cropped_mask, aspect='auto')

            subroi_idx = (i - 1) * (len(subroi_yaxis_borders) - 1) + (j - 1)
            has_flaw[df["subroi_idx"] == subroi_idx] = np.sum(cropped_mask) > (flaw_size * thresh)
    return has_flaw

