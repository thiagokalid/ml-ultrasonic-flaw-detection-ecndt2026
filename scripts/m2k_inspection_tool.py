from framework import file_m2k
from framework.post_proc import envelope
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')

#%%
DATA_ROOT = "../data/m2k/"
FILENAME = "2nd_row_v1.m2k"
# REF_FILENAME = "ref4.m2k"

ith_shot = 0
LOG_CTE = 1E-6
REDUCTION_METHOD = np.median

t_inner = 57.31
t_outer = 62.2
#%%
data_insp = file_m2k.read(DATA_ROOT + FILENAME, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=ith_shot)
# data_ref = file_m2k.read(DATA_ROOT + REF_FILENAME, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
time_grid = data_insp.time_grid[:, 0]
alpha_grid = np.arange(-45, 45 + .5, .5)
print("Hardware gain: ", data_insp.inspection_params.gain_hw)


# Converting channels to S-scan:
channels_insp = data_insp.ascan_data[..., 0]
# channels_ref = data_ref.ascan_data[..., 0]

# channels_list = [channels_insp, channels_ref, channels_insp - channels_ref]
channels_list = [channels_insp, channels_insp, channels_insp]
sscan_list = [REDUCTION_METHOD(channel, axis=2) for channel in channels_list]
sscan_env = [envelope(sscan, axis=0) for sscan in sscan_list]
sscan_norm = [sscan / np.max(sscan_env) for sscan in sscan_env]
sscan_log = [np.log10(sscan + LOG_CTE) for sscan in sscan_norm]
names = ["Inspection", "Reference", "Insp - Ref"]

#%% Plot data:
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
plt.suptitle(f"Experiment name: {FILENAME}")
for i, ith_sscan_log in enumerate(sscan_log):
    ax[i].set_title(names[i])
    ax[i].imshow(ith_sscan_log, extent=[-45, 45, time_grid[-1], time_grid[0]], cmap='inferno', aspect='auto', interpolation='None', vmin=-6, vmax=0)
    ax[i].set_xlabel("$\\alpha$-axis / (degrees)")
    ax[i].set_ylabel("Time / $\\mathrm{\\mu s}$")
    xtmp = np.linspace(-45, 45)
    ytmp = np.ones_like(xtmp)

    # Plot inner and outer surface location:
    plt.plot(xtmp, t_outer * ytmp, ':', color='lime', linewidth=2)
    plt.plot(xtmp, t_inner * ytmp, ':', color='lime', linewidth=2)
plt.tight_layout()

#%%
# --- Plot anomaly bounding boxes ---
df = pd.read_pickle('dataset_creation/acoustic_lens_sscan_dataset_C.pkl')
anomaly_in_insp = df["filename"] == FILENAME
anomaly_in_shot = df['contain_flaw']
anomaly_mask = np.logical_and(anomaly_in_insp, anomaly_in_shot)
if np.any(anomaly_mask):
    borders = df[anomaly_mask]['sscan_limits']
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

plt.show()
