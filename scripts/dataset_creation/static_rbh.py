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
        if value.get("inspection type") == "static rbh"
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

        number_of_flaws = inspection_info[filename]["# flaws"]

        # Apply -6 dB mask to extract points that belong to the flaw
        thresh = 10 ** (-6 / 20)

        # Splitting the S-scan into smallers S-scans:
        subroi_xaxis_borders = np.linspace(-45, 45, NUM_SUBROIS_XAXIS)
        subroi_yaxis_borders = np.linspace(t_outer, t_inner, NUM_SUBROIS_YAXIS)

        global_mask = np.zeros_like(sscan_envelope, dtype=bool)

        for ith_flaw in range(number_of_flaws):
            angular_limits = inspection_info[filename]["ang_flaw"][ith_flaw]
            time_limits = inspection_info[filename]["t_flaw"][ith_flaw]

            row_beg = np.searchsorted(time_grid, time_limits[0])
            row_end = np.searchsorted(time_grid, time_limits[1])
            col_beg = np.searchsorted(alpha_grid, angular_limits[0])
            col_end = np.searchsorted(alpha_grid, angular_limits[1])

            sscan_flaw = sscan_envelope[row_beg:row_end, col_beg:col_end]
            flaw_mask = sscan_flaw >= sscan_flaw.max() * thresh
            global_mask[row_beg:row_end, col_beg:col_end] |= flaw_mask


        # Now loop over sub-ROIs:
        contain_flaw = False
        for i in range(1, len(subroi_xaxis_borders)):
            for j in range(1, len(subroi_yaxis_borders)):
                x_start, x_end = subroi_xaxis_borders[i - 1], subroi_xaxis_borders[i]
                y_start, y_end = subroi_yaxis_borders[j - 1], subroi_yaxis_borders[j]

                t_start_idx = np.searchsorted(time_grid, y_start)
                t_end_idx = np.searchsorted(time_grid, y_end)
                a_start_idx = np.searchsorted(alpha_grid, x_start)
                a_end_idx = np.searchsorted(alpha_grid, x_end)

                sub_sscan = sscan_envelope[t_start_idx:t_end_idx, a_start_idx:a_end_idx]

                flattened_idx = (i - 1) * (len(subroi_yaxis_borders) - 1) + (j - 1)
                roi_mask = global_mask[t_start_idx:t_end_idx, a_start_idx:a_end_idx]

                if np.any(roi_mask):
                    total_pxs = np.sum(global_mask)
                    subroi_pxs = np.sum(roi_mask)

                    contain_flaw = (subroi_pxs / total_pxs) > 0.075
                else:
                    contain_flaw = False

                row = (
                    filename,
                    ref_filename,
                    time_grid[1] - time_grid[0],
                    0.5,
                    ith_shot,
                    flattened_idx,
                    t_outer,
                    t_inner,
                    sscan_max,
                    contain_flaw,
                    [(x_start, x_end), (y_start, y_end)],
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
df.to_pickle(PKL_PATH + "acoustic_lens_sscan_dataset_C.pkl")
