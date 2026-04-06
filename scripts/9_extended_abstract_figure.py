import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from framework import file_m2k
from framework.post_proc import envelope
import joblib
from utils import sscan2tiles, tiles2sscan

linewidth = 6.2

PLOT_DATA = True
DEBUG_PLOT = True
DATA_ROOT = Path("../data/")
DATASET_PATH = DATA_ROOT / "dataset"
MODELS_PATH = DATA_ROOT / "models"
US_DATA = DATA_ROOT / "us_dataset"

LOG_CTE = 1e-6
model = 'lof'
REDUCTION_METHOD = np.median
NUM_tilesS_YAXIS = 10
NUM_tilesS_XAXIS = 20
curr_shot = 178
FILENAME = [
"TuboNovo_260shots.m2k",
#     "passive_dir_0degree_v3.m2k",
#     "passive_dir_10degree_v2.m2k",
#     "passive_dir_20degree_v2.m2k",
    # "passive_dir_30degree_v2.m2k",
    # "passive_dir_34degree_v2.m2k",
    # "passive_dir_38.5degree_v2.m2k"
    # "passive_dir_sweep_bh_v1.m2k"
    # "active_dir_xl_focused_1degree_v1.m2k"
]

# Load inspection info
with open('../data/configs/inspection_info.json', 'r') as f:
    inspection_info = json.load(f)
inspection_info = {fname: inspection_info[fname] for fname in FILENAME if fname in inspection_info}

# Load predictions and scores
predictions_df = pd.read_pickle(MODELS_PATH / model / "prediction_df_show.pkl")
y_scores = joblib.load(MODELS_PATH / model / "y_scores_show.pkl")


for fname in FILENAME:
    info = inspection_info[fname]
    num_shots = info["number_of_shots"]
    t_outer, t_inner = info['surface_position']
    tiles_xaxis_borders = np.linspace(-45, 45, NUM_tilesS_XAXIS)
    tiles_yaxis_borders = np.linspace(t_outer, t_inner, NUM_tilesS_YAXIS)



    # --- Retrieve data ---
    data_insp = file_m2k.read(
        str(US_DATA / fname),
        freq_transd=5,
        bw_transd=0.5,
        tp_transd='gaussian',
        sel_shots=curr_shot
    )
    time_grid = data_insp.time_grid
    channels_insp = data_insp.ascan_data[..., 0]
    sscan = REDUCTION_METHOD(channels_insp, axis=2)
    sscan_envelope = envelope(sscan, axis=0)
    sscan_log = np.log10(sscan_envelope / sscan_envelope.max() + LOG_CTE)

    # --- Threshold-based method:
    sscan_roi_env = sscan_envelope[np.searchsorted(time_grid[:, 0], t_outer):np.searchsorted(time_grid[:, 0], t_inner), :]
    max_roi = np.max(
        sscan_roi_env
    )

    sscan_mask = sscan_roi_env > max_roi * .25
    alpha_span = np.arange(-45, 45, .5)
    tiles, tiles_idx, limits = sscan2tiles(sscan_envelope, time_grid[:, 0], alpha_span, t_outer, t_inner, 20, 10)
    guesses = np.array([len(patch[patch >= max_roi * .25]) / len(patch) >= 25/100 for patch in tiles])
    anomalies_idx = tiles_idx[guesses]
    limits = np.array(limits)[guesses]


    # binary_tiles = [np.ones_like(patch) * guess for patch, guess in zip(tiles, guessses)]
    # sscan = tiles2sscan(binary_tiles, time_grid, alpha_span, t_outer, t_inner, 20, 10)
    #
    # fig, ax = plt.subplots(figsize=(linewidth * .6, 3))
    #
    # ax.imshow(
    #     sscan,
    #     extent=[-45, 45, time_grid[-1], time_grid[0]],
    #     cmap='inferno',
    #     aspect='auto',
    #     interpolation='none'
    # )

    # --- Prepare heatmap from scores ---
    current_mask = (predictions_df["filename"] == fname) & (predictions_df["shot"] == curr_shot)
    normalize_factor = [(np.min(y_scores[predictions_df["tiles_idx"] == idx]), np.max(y_scores[predictions_df["tiles_idx"] == idx])) for idx in np.unique(predictions_df["tiles_idx"])]

    heatmap = np.zeros_like(sscan_log)
    if np.any(current_mask):
        borders = predictions_df[current_mask]['tiles_limits']
        tiles_idx = predictions_df[current_mask]['tiles_idx']
        scores = y_scores[current_mask]
        x_line = np.linspace(-45, 45, heatmap.shape[1])
        y_line = time_grid
        for ii, (border, score) in enumerate(zip(borders, scores)):
            (xbeg, xend), (zbeg, zend) = border
            xbeg, xend = np.degrees(xbeg), np.degrees(xend)
            mask_x = (x_line >= xbeg) & (x_line <= xend)
            mask_y = (y_line >= zbeg) & (y_line <= zend)
            mask = mask_y[:, None] & mask_x[None, :]
            curr_tiles = tiles_idx.iloc[ii]
            heatmap += ((score - normalize_factor[curr_tiles][0]) / (normalize_factor[curr_tiles][1] - normalize_factor[curr_tiles][0])) * mask[:, 0, :]

        # heatmap = heatmap / (heatmap.max() + 1e-8)

    fig, ax = plt.subplots(figsize=(linewidth*.6, 3))

    ax.imshow(
        sscan_log,
        vmax=0,
        vmin=-6,
        extent=[-45, 45, time_grid[-1], time_grid[0]],
        cmap='gray',
        aspect='auto',
        interpolation='none'
    )
    ax.plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), '--', color='k', linewidth=2)
    ax.plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), '--', color='k', linewidth=2)
    ax.set_xticks(tiles_xaxis_borders)
    ax.set_yticks(tiles_yaxis_borders)
    ax.set_yticklabels([f"{y:.1f}" for y in tiles_yaxis_borders])
    ax.grid(color='k', alpha=0.25)
    # ax.set_title("S-scan with Predicted Anomalies")

    # Determine 4 negative and 4 positive ticks
    x = np.array(tiles_xaxis_borders)
    num_ticks_each_side = 4
    x_ticks_neg = tiles_xaxis_borders[:10:2]  # 4 negatives
    x_ticks_pos = x_ticks_neg[::-1] * -1
    selected_ticks = np.concatenate([x_ticks_neg, x_ticks_pos])

    # Build tick labels: show only near the selected tick positions
    ax.set_xticklabels([
        f"{xv:.0f}" if np.isclose(xv, selected_ticks, atol=1e-6).any() else ""
        for xv in tiles_xaxis_borders
    ])
    ax.set_xlabel(r"$\alpha$-axis / (degrees)")
    ax.set_ylabel(r"Time / ($\mathrm{\mu s}$)")
    # ax.set_ylim([64, 55])
    ax.set_ylim([t_inner + 3, t_outer - 2])
    # Plot anomaly bounding boxes on S-scan
    anomaly_mask = current_mask & (predictions_df["y_pred"] == 1)

    if np.any(anomaly_mask):
        # ---- 1) Collect boxes (convert to degrees only for x) ----
        boxes = []
        for (xbeg, xend), (zbeg, zend) in predictions_df[anomaly_mask]['tiles_limits']:
            boxes.append([np.degrees(xbeg), np.degrees(xend), zbeg, zend])

        boxes = np.array(boxes)  # shape (N, 4)

        # ---- 2) Sort by xbeg for stable merging ----
        boxes = boxes[np.argsort(boxes[:, 0])]

        # ---- 3) Merge neighbors (touching or overlapping intervals) ----
        merged = []
        cur = boxes[0].copy()

        for box in boxes[1:]:
            xbeg, xend, zbeg, zend = box

            # If boxes overlap or touch in BOTH x and z:
            touch_x = xbeg <= cur[1]  # overlapping/touching x-range
            touch_z = not (zend < cur[2] or zbeg > cur[3])  # overlapping/touching z-range

            if touch_x and touch_z:
                # Expand current merged box
                cur[0] = min(cur[0], xbeg)
                cur[1] = max(cur[1], xend)
                cur[2] = min(cur[2], zbeg)
                cur[3] = max(cur[3], zend)
            else:
                merged.append(cur.copy())
                cur = box.copy()

        merged.append(cur)  # don't forget last one

        # ---- 4) Plot merged boxes ----
        for i, (xbeg, xend, zbeg, zend) in enumerate(merged):
            label = "Proposed" if i == 0 else "_"

            ax.plot(
                [xbeg, xend, xend, xbeg, xbeg],
                [zbeg, zbeg, zend, zend, zbeg],
                color='b',
                alpha=0.75,
                label=label,
                lw=3,
            )

        # %%
        if np.any(guesses):
            # ---- 1) Collect boxes (convert to degrees only for x) ----
            boxes = []
            for (xbeg, xend), (zbeg, zend) in limits:
                boxes.append([xbeg, xend, zbeg, zend])

            boxes = np.array(boxes)  # shape (N, 4)

            # ---- 2) Sort by xbeg for stable merging ----
            boxes = boxes[np.argsort(boxes[:, 0])]

            # ---- 3) Merge neighbors (touching or overlapping intervals) ----
            merged = []
            cur = boxes[0].copy()

            for box in boxes[1:]:
                xbeg, xend, zbeg, zend = box

                # If boxes overlap or touch in BOTH x and z:
                touch_x = xbeg <= cur[1]  # overlapping/touching x-range
                touch_z = not (zend < cur[2] or zbeg > cur[3])  # overlapping/touching z-range

                if touch_x and touch_z:
                    # Expand current merged box
                    cur[0] = min(cur[0], xbeg)
                    cur[1] = max(cur[1], xend)
                    cur[2] = min(cur[2], zbeg)
                    cur[3] = max(cur[3], zend)
                else:
                    merged.append(cur.copy())
                    cur = box.copy()

            merged.append(cur)  # don't forget last one

            # ---- 4) Plot merged boxes ----
            for i, (xbeg, xend, zbeg, zend) in enumerate(merged):
                label = "Threshold-based" if i == 0 else "_"

                ax.plot(
                    [xbeg, xend, xend, xbeg, xbeg],
                    [zbeg, zbeg, zend, zend, zbeg],
                    color='lime',
                    alpha=0.75,
                    label=label,
                )

    #%%
    # ax.legend(loc="lower right")

    # Right: Heatmap of predicted flaw scores
    # im = axes[1].imshow(
    #     heatmap,
    #     extent=[-45, 45, time_grid[-1], time_grid[0]],
    #     cmap='jet',
    #     aspect='auto',
    #     interpolation='none',
    #     vmin=0,
    #     vmax=1
    # )
    # axes[1].plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
    # axes[1].plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
    # axes[1].set_xticks(tiles_xaxis_borders)
    # axes[1].set_yticks(tiles_yaxis_borders)
    # axes[1].set_yticklabels([f"{y:.1f}" for y in tiles_yaxis_borders])
    # axes[1].grid(color='k', alpha=0.25)
    # axes[1].set_title("Predicted Flaw Score Heatmap (Normalized per patch)")
    # axes[1].set_xlabel(r"$\alpha$-axis / (degrees)")
    # axes[1].set_ylabel(r"Time / ($\mathrm{\mu s}$)")
    #
    # # Determine 4 negative and 4 positive ticks
    # x = np.array(tiles_xaxis_borders)
    # num_ticks_each_side = 4
    # x_ticks_neg = tiles_xaxis_borders[:10:2]  # 4 negatives
    # x_ticks_pos = x_ticks_neg[::-1] * -1
    # selected_ticks = np.concatenate([x_ticks_neg, x_ticks_pos])
    #
    # # Build tick labels: show only near the selected tick positions
    # axes[1].set_xticklabels([
    #     f"{xv:.0f}" if np.isclose(xv, selected_ticks, atol=1e-6).any() else ""
    #     for xv in tiles_xaxis_borders
    # ])
    #
    # axes[1].set_yticklabels([
    #     f"{y:.1f}" if i % 2 == 0 else "" for i, y in enumerate(tiles_yaxis_borders)
    # ])
    # fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    #%%
    # Right: Heatmap of predicted flaw scores
    import re
    import cv2

    filename = fname
    m2k_filename = filename[:filename.find("m2k") - 1] + ".m2k"
    m2k_filename = re.sub(r"\.", "_", m2k_filename)
    m2k_filename = f"{m2k_filename}_shot{curr_shot}_"

    # List all PNGs in the directory
    mask_dir = "../data/masks/"
    png_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    # Find the one that starts with m2k_filename
    matching_file = next((f for f in png_files if f.startswith(m2k_filename)), None)

    if matching_file:
        mask_path = os.path.join(mask_dir, matching_file)
        print("Found:", mask_path)
        # Read image file in grayscale as numpy array
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        print("No matching mask file found.")
        # Create zero array with same shape as sscan_log
        mask = np.zeros_like(sscan_log, dtype=np.uint8)

    true_df = pd.read_pickle(DATASET_PATH / "dataset.pkl")
    current_mask = (true_df["filename"] == fname) & (true_df["shot"] == curr_shot)
    anomaly_mask = current_mask & (true_df["contain_flaw"] == 1)

    if np.any(anomaly_mask):
        # ---- 1) Collect boxes (convert to degrees only for x) ----
        boxes = []
        for (xbeg, xend), (zbeg, zend) in true_df[anomaly_mask]['tiles_limits']:
            boxes.append([np.degrees(xbeg), np.degrees(xend), zbeg, zend])

        boxes = np.array(boxes)  # shape (N, 4)

        # ---- 2) Sort by xbeg for stable merging ----
        boxes = boxes[np.argsort(boxes[:, 0])]

        # ---- 3) Merge neighbors (touching or overlapping intervals) ----
        merged = []
        cur = boxes[0].copy()

        for box in boxes[1:]:
            xbeg, xend, zbeg, zend = box

            # If boxes overlap or touch in BOTH x and z:
            touch_x = xbeg <= cur[1]  # overlapping/touching x-range
            touch_z = not (zend < cur[2] or zbeg > cur[3])  # overlapping/touching z-range

            if touch_x and touch_z:
                # Expand current merged box
                cur[0] = min(cur[0], xbeg)
                cur[1] = max(cur[1], xend)
                cur[2] = min(cur[2], zbeg)
                cur[3] = max(cur[3], zend)
            else:
                merged.append(cur.copy())
                cur = box.copy()

        merged.append(cur)  # don't forget last one

        # ---- 4) Plot merged boxes ----
        for i, (xbeg, xend, zbeg, zend) in enumerate(merged):
            label = "Ground-truth" if i == 0 else "_"

            ax.plot(
                [xbeg, xend, xend, xbeg, xbeg],
                [zbeg, zbeg, zend, zend, zbeg],
                color='r',
                alpha=0.75,
                label=label,
            )

    plt.legend(ncols=2, loc='lower center', handletextpad=0.1, columnspacing=0, fancybox=False)
    plt.tight_layout()
    plt.savefig("../figures/sscan_anomalies.pdf", dpi=300)
    plt.show()
    #%%


    fig, ax = plt.subplots(figsize=(linewidth*.6, 3))
    im = ax.imshow(
        mask,
        extent=[-45, 45, t_inner, t_outer],
        cmap='binary',
        aspect='auto',
        interpolation='none',
        vmin=0,
        vmax=1
    )
    ax.plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), '--', color='k', linewidth=2)
    ax.plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), '--', color='k', linewidth=2)
    ax.set_xticks(tiles_xaxis_borders)
    ax.set_yticks(tiles_yaxis_borders)
    ax.set_yticklabels([f"{y:.1f}" for y in tiles_yaxis_borders])
    ax.grid(color='k', alpha=0.25)
    # ax.set_title("Flaw Position Ground-truth")
    ax.set_xlabel(r"$\alpha$-axis / (degrees)")
    ax.set_ylabel(r"Time / ($\mathrm{\mu s}$)")
    ax.set_ylim([t_inner + 2, t_outer - 2])

    # Determine 4 negative and 4 positive ticks
    x = np.array(tiles_xaxis_borders)
    num_ticks_each_side = 4
    x_ticks_neg = tiles_xaxis_borders[:10:2]  # 4 negatives
    x_ticks_pos = x_ticks_neg[::-1] * -1
    selected_ticks = np.concatenate([x_ticks_neg, x_ticks_pos])

    # Build tick labels: show only near the selected tick positions
    ax.set_xticklabels([
        f"{xv:.0f}" if np.isclose(xv, selected_ticks, atol=1e-6).any() else ""
        for xv in tiles_xaxis_borders
    ])

    ax.set_yticklabels([
        f"{y:.1f}" if i % 2 == 0 else "" for i, y in enumerate(tiles_yaxis_borders)
    ])

    #%%

    true_df = pd.read_pickle(DATASET_PATH / "dataset.pkl")
    current_mask = (true_df["filename"] == fname) & (true_df["shot"] == curr_shot)
    anomaly_mask = current_mask & (true_df["contain_flaw"] == 1)

    if np.any(anomaly_mask):
        # ---- 1) Collect boxes (convert to degrees only for x) ----
        boxes = []
        for (xbeg, xend), (zbeg, zend) in true_df[anomaly_mask]['tiles_limits']:
            boxes.append([np.degrees(xbeg), np.degrees(xend), zbeg, zend])

        boxes = np.array(boxes)  # shape (N, 4)

        # ---- 2) Sort by xbeg for stable merging ----
        boxes = boxes[np.argsort(boxes[:, 0])]

        # ---- 3) Merge neighbors (touching or overlapping intervals) ----
        merged = []
        cur = boxes[0].copy()

        for box in boxes[1:]:
            xbeg, xend, zbeg, zend = box

            # If boxes overlap or touch in BOTH x and z:
            touch_x = xbeg <= cur[1]  # overlapping/touching x-range
            touch_z = not (zend < cur[2] or zbeg > cur[3])  # overlapping/touching z-range

            if touch_x and touch_z:
                # Expand current merged box
                cur[0] = min(cur[0], xbeg)
                cur[1] = max(cur[1], xend)
                cur[2] = min(cur[2], zbeg)
                cur[3] = max(cur[3], zend)
            else:
                merged.append(cur.copy())
                cur = box.copy()

        merged.append(cur)  # don't forget last one

        # ---- 4) Plot merged boxes ----
        for i, (xbeg, xend, zbeg, zend) in enumerate(merged):
            label = "Proposed." if i == 0 else "_"

            ax.plot(
                [xbeg, xend, xend, xbeg, xbeg],
                [zbeg, zbeg, zend, zend, zbeg],
                color='r',
                alpha=0.75,
                label=label,
            )

    #%%

    # --- Save figure ---
    folder_root = f"../figures/{fname}_inspection"
    os.makedirs(folder_root, exist_ok=True)
    plt.tight_layout()
    plt.savefig("../figures/binary_mask_anomalies.pdf", dpi=300)
    # plt.savefig(f"{folder_root}/img_{curr_shot:02d}.png", dpi=300)
    # plt.close(fig)
    plt.show()
