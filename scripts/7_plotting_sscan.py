import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from framework import file_m2k
from framework.post_proc import envelope
import joblib

PLOT_DATA = True
DEBUG_PLOT = True
DATA_ROOT = "../data/m2k/"
PKL_DATA_PATH = "../data/pkl/"
MODEL_PATH = PKL_DATA_PATH + "models/"
LOG_CTE = 1e-6
model = 'lof'
REDUCTION_METHOD = np.median
NUM_SUBROIS_YAXIS = 10
NUM_SUBROIS_XAXIS = 20
FILENAME = [
    # "TuboNovo_260shots.m2k",
    "passive_dir_0degree_v2.m2k",
    # "passive_dir_10degree_v2.m2k",
    # "passive_dir_20degree_v2.m2k",
    # "passive_dir_30degree_v2.m2k",
    # "passive_dir_34degree_v2.m2k",
    # "passive_dir_sweep_bh_v2.m2k"
    # "active_dir_xl_focused_1degree_v1.m2k"
]

# Load inspection info
with open('../data/configs/inspection_info.json', 'r') as f:
    inspection_info = json.load(f)
inspection_info = {fname: inspection_info[fname] for fname in FILENAME if fname in inspection_info}

# Load predictions and scores
predictions_df = pd.read_pickle(MODEL_PATH + model + "_prediction_df_show.pkl")
y_scores = joblib.load(MODEL_PATH + model + "_y_scores_show.pkl")
y_combined = joblib.load(PKL_DATA_PATH + "y_combined.pkl")

plt.ioff()  # Disable interactive mode

for fname in FILENAME:
    info = inspection_info[fname]
    num_shots = info["number_of_shots"]
    t_outer, t_inner = info['surface_position']
    subroi_xaxis_borders = np.linspace(-45, 45, NUM_SUBROIS_XAXIS)
    subroi_yaxis_borders = np.linspace(t_outer, t_inner, NUM_SUBROIS_YAXIS)

    for curr_shot in tqdm(range(num_shots), desc=f"Processing {fname}"):
        # --- Prepare heatmap from scores ---
        current_mask = (predictions_df["filename"] == fname) & (predictions_df["shot"] == curr_shot)

        # --- Retrieve data ---
        data_insp = file_m2k.read(
            DATA_ROOT + fname,
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

        # --- Prepare heatmap from scores ---
        current_mask = (predictions_df["filename"] == fname) & (predictions_df["shot"] == curr_shot)
        normalize_factor = [(np.min(y_scores[predictions_df["subroi_idx"] == idx]), np.max(y_scores[predictions_df["subroi_idx"] == idx])) for idx in np.unique(predictions_df["subroi_idx"])]

        heatmap = np.zeros_like(sscan_log)
        if np.any(current_mask):
            borders = predictions_df[current_mask]['subroi_limits']
            subroi_idx = predictions_df[current_mask]['subroi_idx']
            scores = y_scores[current_mask]
            x_line = np.linspace(-45, 45, heatmap.shape[1])
            y_line = time_grid
            for ii, (border, score) in enumerate(zip(borders, scores)):
                (xbeg, xend), (zbeg, zend) = border
                xbeg, xend = np.degrees(xbeg), np.degrees(xend)
                mask_x = (x_line >= xbeg) & (x_line <= xend)
                mask_y = (y_line >= zbeg) & (y_line <= zend)
                mask = mask_y[:, None] & mask_x[None, :]
                curr_subroi = subroi_idx.iloc[ii]
                heatmap += ((score - normalize_factor[curr_subroi][0]) / (normalize_factor[curr_subroi][1] - normalize_factor[curr_subroi][0])) * mask[:, 0, :]

            # heatmap = heatmap / (heatmap.max() + 1e-8)

        # --- Plot side by side ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(f"Inspection: {fname}; Shot: {curr_shot}")

        # Left: S-scan with bounding boxes
        axes[0].imshow(
            sscan_log,
            vmax=0,
            vmin=-6,
            extent=[-45, 45, time_grid[-1], time_grid[0]],
            cmap='inferno',
            aspect='auto',
            interpolation='none'
        )
        axes[0].plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
        axes[0].plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
        axes[0].set_xticks(subroi_xaxis_borders)
        axes[0].set_yticks(subroi_yaxis_borders)
        axes[0].set_yticklabels([f"{y:.1f}" for y in subroi_yaxis_borders])
        axes[0].grid(color='k', alpha=0.25)
        axes[0].set_title("S-scan with Predicted Anomalies")

        # Determine 4 negative and 4 positive ticks
        x = np.array(subroi_xaxis_borders)
        num_ticks_each_side = 4
        x_ticks_neg = subroi_xaxis_borders[:10:2]  # 4 negatives
        x_ticks_pos = x_ticks_neg[::-1] * -1
        selected_ticks = np.concatenate([x_ticks_neg, x_ticks_pos])

        # Build tick labels: show only near the selected tick positions
        axes[0].set_xticklabels([
            f"{xv:.0f}" if np.isclose(xv, selected_ticks, atol=1e-6).any() else ""
            for xv in subroi_xaxis_borders
        ])
        axes[0].set_xlabel(r"$\alpha$-axis / (degrees)")
        axes[0].set_ylabel(r"Time / ($\mathrm{\mu s}$)")

        # Plot anomaly bounding boxes on S-scan
        anomaly_mask = current_mask & (predictions_df["y_pred"] == 1)
        if np.any(anomaly_mask):
            for ii, border in enumerate(predictions_df[anomaly_mask]['subroi_limits']):
                (xbeg, xend), (zbeg, zend) = border
                xbeg, xend = np.degrees(xbeg), np.degrees(xend)
                label = "Anomaly" if ii == 0 else "_"
                axes[0].plot(
                    [xbeg, xend, xend, xbeg, xbeg],
                    [zbeg, zbeg, zend, zend, zbeg],
                    color='b',
                    alpha=0.5,
                    label=label
                )
        axes[0].legend()

        # Right: Heatmap of predicted flaw scores
        im = axes[1].imshow(
            heatmap,
            extent=[-45, 45, time_grid[-1], time_grid[0]],
            cmap='jet',
            aspect='auto',
            interpolation='none',
            vmin=0,
            vmax=1
        )
        axes[1].plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
        axes[1].plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
        axes[1].set_xticks(subroi_xaxis_borders)
        axes[1].set_yticks(subroi_yaxis_borders)
        axes[1].set_yticklabels([f"{y:.1f}" for y in subroi_yaxis_borders])
        axes[1].grid(color='k', alpha=0.25)
        axes[1].set_title("Predicted Flaw Score Heatmap (Normalized per patch)")
        axes[1].set_xlabel(r"$\alpha$-axis / (degrees)")
        axes[1].set_ylabel(r"Time / ($\mathrm{\mu s}$)")

        # Determine 4 negative and 4 positive ticks
        x = np.array(subroi_xaxis_borders)
        num_ticks_each_side = 4
        x_ticks_neg = subroi_xaxis_borders[:10:2]  # 4 negatives
        x_ticks_pos = x_ticks_neg[::-1] * -1
        selected_ticks = np.concatenate([x_ticks_neg, x_ticks_pos])

        # Build tick labels: show only near the selected tick positions
        axes[1].set_xticklabels([
            f"{xv:.0f}" if np.isclose(xv, selected_ticks, atol=1e-6).any() else ""
            for xv in subroi_xaxis_borders
        ])

        axes[1].set_yticklabels([
            f"{y:.1f}" if i % 2 == 0 else "" for i, y in enumerate(subroi_yaxis_borders)
        ])
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

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


        im = axes[2].imshow(
            mask,
            extent=[-45, 45, t_inner, t_outer],
            cmap='binary',
            aspect='auto',
            interpolation='none',
            vmin=0,
            vmax=1
        )
        axes[2].plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
        axes[2].plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), ':', color='lime', linewidth=2)
        axes[2].set_xticks(subroi_xaxis_borders)
        axes[2].set_yticks(subroi_yaxis_borders)
        axes[2].set_yticklabels([f"{y:.1f}" for y in subroi_yaxis_borders])
        axes[2].grid(color='k', alpha=0.25)
        axes[2].set_title("Flaw Position Ground-truth")
        axes[2].set_xlabel(r"$\alpha$-axis / (degrees)")
        axes[2].set_ylabel(r"Time / ($\mathrm{\mu s}$)")
        axes[2].set_ylim([time_grid[-1], time_grid[0]])

        # Determine 4 negative and 4 positive ticks
        x = np.array(subroi_xaxis_borders)
        num_ticks_each_side = 4
        x_ticks_neg = subroi_xaxis_borders[:10:2]  # 4 negatives
        x_ticks_pos = x_ticks_neg[::-1] * -1
        selected_ticks = np.concatenate([x_ticks_neg, x_ticks_pos])

        # Build tick labels: show only near the selected tick positions
        axes[2].set_xticklabels([
            f"{xv:.0f}" if np.isclose(xv, selected_ticks, atol=1e-6).any() else ""
            for xv in subroi_xaxis_borders
        ])

        axes[2].set_yticklabels([
            f"{y:.1f}" if i % 2 == 0 else "" for i, y in enumerate(subroi_yaxis_borders)
        ])
        #%%
        plt.tight_layout()
        plt.show(block=True)

        # # --- Save figure ---
        # folder_root = f"../figures/{fname}_inspection"
        # os.makedirs(folder_root, exist_ok=True)
        # plt.tight_layout()
        # plt.savefig(f"{folder_root}/img_{curr_shot:02d}.png", dpi=300)
        # plt.close(fig)

plt.ion()
plt.close('all')
