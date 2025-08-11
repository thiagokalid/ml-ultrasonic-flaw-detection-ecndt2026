import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from framework import file_m2k
from framework.post_proc import envelope

PLOT_DATA = False
DEBUG_PLOT = False
DATA_ROOT = "../../data/m2k/"
PKL_PATH = "../../data/pkl/"
LOG_CTE = 1e-6
REDUCTION_METHOD = np.median
NUM_SUBROIS_YAXIS = 10
NUM_SUBROIS_XAXIS = 20


data = pd.DataFrame()


with open('../inspection_info.json', 'r') as f:
    raw_inspection_info = json.load(f)
    inspection_info = {
        key: value
        for key, value in raw_inspection_info.items()
        if value.get("inspection type") == "active sweep sdh"
    }

selected_keys = [
"active_dir_xl_focused_1degree_v1.m2k",
"active_dir_xl_focused_1degree_v2.m2k",
"active_dir_xl_focused_1degree_v3.m2k"
]

inspection_info = {
    key: inspection_info[key]
    for key in selected_keys
}

dataframe_rows = []
for filename in tqdm(inspection_info.keys()):
    ref_filename = ""
    for ith_shot in range(inspection_info[filename]["number_of_shots"]):
        # Reading the US data:
        data_insp = file_m2k.read(DATA_ROOT + filename, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=ith_shot)
        time_grid = data_insp.time_grid[:, 0]
        alpha_grid = np.arange(-45, 45 + .5, .5)


        # Converting channels to S-scan:
        channels_insp = data_insp.ascan_data[..., 0]
        sscan = REDUCTION_METHOD(channels_insp, axis=2)
        sscan_envelope = envelope(sscan, axis=0)
        sscan_max = sscan_envelope.max()

        # Extract outer and inner surface manually estimated TOF:
        t_outer, t_inner = inspection_info[filename]["surface_position"]

        # Apply -6 dB mask to extract points that belong to the flaw
        thresh = 10 ** (-6 / 20)

        # Flaw ROI:
        step = ith_shot * 1
        ang_beg, ang_end = -7 + step, 10 + step
        t_beg, t_end = inspection_info[filename]["t_flaw"]
        row_beg, row_end, col_beg, col_end = np.searchsorted(time_grid, t_beg), np.searchsorted(time_grid, t_end), np.searchsorted( alpha_grid, ang_beg), np.searchsorted(alpha_grid, ang_end)
        sscan_flaw = sscan_envelope[row_beg:row_end, col_beg:col_end]
        flaw_mask = sscan_flaw >= sscan_flaw.max() * thresh

        global_mask = np.zeros_like(sscan_envelope, dtype=bool)
        global_mask[row_beg:row_end, col_beg:col_end] = flaw_mask

        # Splitting the S-scan into smallers S-scans:
        subroi_xaxis_borders = np.linspace(-45, 45, NUM_SUBROIS_XAXIS)
        subroi_yaxis_borders = np.linspace(t_outer, t_inner, NUM_SUBROIS_YAXIS)

        # Loop over sub-ROIs
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

                sub_sscan = sscan_envelope[t_start_idx:t_end_idx, a_start_idx:a_end_idx]

                # Check if there is a flaw inside the sub-roi
                if np.any(global_mask[t_start_idx:t_end_idx, a_start_idx:a_end_idx] == True):
                    total_pxs = np.sum(global_mask)
                    subroi_pxs = np.sum(global_mask[t_start_idx:t_end_idx, a_start_idx:a_end_idx])
                    if subroi_pxs / total_pxs > .1:
                        contain_flaw = True
                    else:
                        contain_flaw = False
                else:
                    contain_flaw = False

                row = (
                    filename,
                    ref_filename,
                    time_grid[1] - time_grid[0],
                    0.5,
                    ith_shot,
                    (i - 1) * (len(subroi_yaxis_borders) - 1) + (j - 1),
                    t_outer,
                    t_inner,
                    sscan_max,
                    contain_flaw,
                    sscan_limits,
                    sub_sscan
                )


                dataframe_rows.append(row)

columns = [
    'filename',
    'ref_filename',
    'dt',
    'dalpha',
    'ith_shot',
    'subroi_idx',
    't_outer',
    't_inner',
    'sscan_max',
    'contain_flaw',
    'sscan_limits',
    'sub_sscan'
]

df = pd.DataFrame(dataframe_rows, columns=columns)
df.to_pickle(PKL_PATH + "acoustic_lens_sscan_dataset_A.pkl")
