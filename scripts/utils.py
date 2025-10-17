from framework.post_proc import envelope as envelope_fun
from framework import file_m2k
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def apply_preprocessing(sscan, envelope: bool, normalize: str, **kwargs) -> np.ndarray:
    new_sscan = np.copy(sscan)

    if envelope:
        new_sscan = envelope_fun(new_sscan, axis=0)

    match normalize:
        case "minmax":
            new_sscan = (new_sscan - new_sscan.min()) / (new_sscan.max() - new_sscan.min())
        case "hw_gain":
            hw_gain_db = kwargs["hw_gain_db"]
            hw_gain_lin = 10**(hw_gain_db/20) # Assuming that A_db = 20 * log10(A_linear)
            new_sscan = 1/hw_gain_lin * new_sscan
        case "None":
            pass
        case _:
            raise ValueError(f"Unknown normalization {normalize}")

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


def split_sscan2subrois(df_rows, sscan: np.ndarray, shot, time_grid: np.ndarray, alpha_grid: np.ndarray, t_outer, t_inner, params, sscan_hog=None) -> None:
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

            if sscan_hog is not None:
                df_rows["sub_sscan_hog"].append(sscan_hog[t_start_idx:t_end_idx, a_start_idx:a_end_idx])
            else:
                df_rows["sub_sscan_hog"].append(None)

    return None

def mark_subsscans(df, mask_resized, time_grid, alpha_grid, t_outer, t_inner, params, thresh=1/100):
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
            has_flaw[df["subroi_idx"] == subroi_idx] = (has_flaw[df["subroi_idx"] == subroi_idx]) or (np.sum(cropped_mask) > (flaw_size * thresh))
    return has_flaw


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        row_sums = cf.sum(axis=1, keepdims=True)
        normalized = cf / row_sums
        group_percentages = ["{0:.2%}".format(value) for value in normalized.flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)