import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from framework import file_m2k
from framework.post_proc import envelope

PLOT_DATA = True
DEBUG_PLOT = True
DATA_ROOT = "../data/m2k/"
PKL_DATA_PATH = "../data/pkl/"
LOG_CTE = 1e-6
model = 'lof'
REDUCTION_METHOD = np.median
NUM_SUBROIS_YAXIS = 10
NUM_SUBROIS_XAXIS = 20

# Test files:
FILENAME = [
"2nd_row_v2.m2k",
"3rd_row_v2.m2k",
"passive_dir_0degree_v2.m2k",
"passive_dir_10degree_v2.m2k",
# "passive_dir_20degree_v2.m2k",
# "passive_dir_30degree_v2.m2k",
# "passive_dir_34degree_v2.m2k",
# "passive_dir_38.5degree_v2.m2k",
# "active_dir_xl_focused_1degree_v1.m2k",
]

data = pd.DataFrame()

with open('../data/m2k/inspection_info.json', 'r') as f:
    inspection_info = json.load(f)

inspection_info = {
    fname: inspection_info[fname]
    for fname in FILENAME
    if fname in inspection_info
}

anomaly_df = pd.read_pickle(PKL_DATA_PATH + model + "_anomalies_df.pkl")

#%%
plt.ioff()  # Already disables interactive mode

for fname in FILENAME:
    info = inspection_info[fname]
    num_shots = info["number_of_shots"]
    ref_filename = info["ref_filename"]
    t_outer, t_inner = info['surface_position']
    subroi_xaxis_borders = np.linspace(-45, 45, NUM_SUBROIS_XAXIS)
    subroi_yaxis_borders = np.linspace(t_outer, t_inner, NUM_SUBROIS_YAXIS)

    for curr_shot in tqdm(range(num_shots), desc=f"Processing {fname}"):
        plt.ioff()

        # --- Retrieve data ---
        data_insp = file_m2k.read(
            DATA_ROOT + fname,
            freq_transd=5,
            bw_transd=0.5,
            tp_transd='gaussian',
            sel_shots=curr_shot
        )
        data_ref = file_m2k.read(
            DATA_ROOT + ref_filename,
            freq_transd=5,
            bw_transd=0.5,
            tp_transd='gaussian',
            sel_shots=0
        )
        time_grid = data_insp.time_grid

        # --- Convert to S-scan and process ---
        channels_insp = data_insp.ascan_data[..., 0]
        channels_ref = data_ref.ascan_data[..., 0]
        sscan = REDUCTION_METHOD(channels_insp - channels_ref, axis=2)
        sscan_envelope = envelope(sscan, axis=0)
        sscan_log = np.log10(sscan_envelope / sscan_envelope.max() + LOG_CTE)

        # --- Plot ---
        fig = plt.figure(figsize=(12, 6))
        plt.suptitle(f"Inspection: {fname}; Shot: {curr_shot}")
        plt.imshow(
            sscan_log,
            extent=[-45, 45, time_grid[-1], time_grid[0]],
            cmap='inferno',
            aspect='auto',
            interpolation='None'
        )

        xtmp = np.linspace(-45, 45)
        ytmp = np.ones_like(xtmp)
        plt.plot(xtmp, t_outer * ytmp, ':', color='lime', linewidth=2)
        plt.plot(xtmp, t_inner * ytmp, ':', color='lime', linewidth=2)

        plt.xticks(subroi_xaxis_borders)
        plt.yticks(subroi_yaxis_borders)
        plt.grid(color='k', alpha=.25)

        # --- Plot anomaly bounding boxes ---
        anomaly_in_insp = anomaly_df["filename"] == fname
        anomaly_in_shot = anomaly_df["ith_shot"] == curr_shot
        anomaly_mask = np.logical_and(anomaly_in_insp, anomaly_in_shot)
        if np.any(anomaly_mask):
            borders = anomaly_df[anomaly_mask]['sscan_limits']
            for ii, border in enumerate(borders):
                (xbeg, xend), (zbeg, zend) = border
                label = "Anomaly" if ii == 0 else "_"
                plt.plot(
                    [xbeg, xend, xend, xbeg, xbeg],
                    [zbeg, zbeg, zend, zend, zbeg],
                    color='b',
                    alpha=0.5,
                    label=label
                )
        plt.legend()

        # --- Save figure ---
        folder_root = f"../figures/{fname}_inspection"
        os.makedirs(folder_root, exist_ok=True)
        plt.savefig(f"{folder_root}/img_{curr_shot:02d}.png", dpi=300)

plt.ion()
plt.close('all')