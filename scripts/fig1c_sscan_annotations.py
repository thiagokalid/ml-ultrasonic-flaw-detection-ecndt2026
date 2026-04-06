import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from framework import file_m2k
from framework.post_proc import envelope
import joblib

from scripts.utils import plot_tiles_labels
from utils import sscan2tiles, tiles2sscan

linewidth = 6.2


# Relevant paths:
PLOT_DATA = True
DEBUG_PLOT = True
DATA_ROOT = Path("../data/")
DATASET_PATH = DATA_ROOT / "dataset"
MODELS_PATH = DATA_ROOT / "models"
US_DATA = DATA_ROOT / "us_dataset"
FILENAME = "active_dir_xl_focused_1degree_v1.m2k"


# Relevant constants:
LOG_CTE = 1e-6
REDUCTION_METHOD = np.median
NUM_tilesS_YAXIS = 10
NUM_tilesS_XAXIS = 20
curr_shot = 30

# Load inspection info
with open('../data/configs/inspection_info.json', 'r') as f:
    inspection_info = json.load(f)
inspection_info = {FILENAME: inspection_info[FILENAME]}


info = inspection_info[FILENAME]
num_shots = info["number_of_shots"]
t_outer, t_inner = info['surface_position']
tiles_xaxis_borders = np.linspace(-45, 45, NUM_tilesS_XAXIS)
tiles_yaxis_borders = np.linspace(t_outer, t_inner, NUM_tilesS_YAXIS)



# --- Retrieve data ---
data_insp = file_m2k.read(
    str(US_DATA / FILENAME),
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


#%%
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
ax.axhline(t_outer, ls='--', color='k', lw=2)
ax.plot(np.linspace(-45, 45), t_outer * np.ones_like(np.linspace(-45, 45)), '--', color='k', linewidth=2)
ax.plot(np.linspace(-45, 45), t_inner * np.ones_like(np.linspace(-45, 45)), '--', color='k', linewidth=2)
ax.set_xticks(np.arange(-45, 45 + 15, 15))
ax.set_yticks(tiles_yaxis_borders[::2])
ax.set_yticklabels([f"{y:.1f}" for y in tiles_yaxis_borders[::2]])
ax.grid(color='k', alpha=0.125)

ax.set_xlabel(r"$\alpha$-axis / (degrees)")
ax.set_ylabel(r"Time / ($\mathrm{\mu s}$)")
ax.set_ylim([t_inner + 2, t_outer - 2])

ax2 = ax.twinx()
ax2.set_ylim([45, 75])
ax2.set_yticks([50, 55, 60, 65, 70])
ax2.set_ylabel(r"Radius / ($\mathrm{mm}$)")

#
textcolor = 'b'

# =============================================
ax.text(-4, 60.3, "Geometrical Artifacts", color=textcolor, ha="center", va="center", fontweight='bold')
ax.annotate("",
            xy=(-16, 59.2),
            xytext=(-4.4, 59.7),
            arrowprops=dict(arrowstyle="-|>", color=textcolor, linewidth=1.5)
            )
ax.annotate("",
            xy=(-4, 58),
            xytext=(-4.4, 59.7),
            arrowprops=dict(arrowstyle="-|>", color=textcolor, linewidth=1.5)
            )
ax.annotate("",
            xy=(16, 58.5),
            xytext=(-4.4, 59.7),
            arrowprops=dict(arrowstyle="-|>", color=textcolor, linewidth=1.5,)
            )

# =============================================
ax.text(-25, 62.3, "Inner surface", color=textcolor, ha="center", va="center", fontweight='bold')
ax.annotate("",
            xy=(-2, 61.5),
            xytext=(-4, 62.3),
            arrowprops=dict(arrowstyle="-|>", color=textcolor, linewidth=2)
            )

# =============================================
ax.text(-24, 57.3, "Outer surface", color=textcolor, ha="center", va="center", fontweight='bold')
ax.annotate("",
            xy=(-2.0, 56.0),
            xytext=(-5.0, 57.1),
            arrowprops=dict(arrowstyle="-|>", color=textcolor, linewidth=2)
            )

# =============================================
ax.text(20, 57.5, "SDH", color=textcolor, ha="center", va="center", fontweight='bold')
ax.annotate("",
            xy=(30, 58.7),
            xytext=(24.5, 57.7),
            arrowprops=dict(arrowstyle="-|>", color=textcolor, linewidth=2)
            )

plt.tight_layout()
plt.savefig("../figures/sscan_annotations.pdf", dpi=300)
plt.show()