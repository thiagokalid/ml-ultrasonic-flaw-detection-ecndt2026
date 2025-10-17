import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

import numpy as np
from scipy import ndimage
from PIL import Image # For converting NumPy array to image and saving
import tifffile
from tqdm import tqdm
from framework import file_m2k
from framework.post_proc import envelope
from PIL import Image
import numpy as np
import os

PLOT_DATA = False
DEBUG_PLOT = False
DATA_ROOT = "../data/m2k/"
PKL_PATH = "../data/pkl/"

LOG_CTE = 1e-6
REDUCTION_METHOD = np.median
NUM_SUBROIS_YAXIS = 10
NUM_SUBROIS_XAXIS = 20
IMG_SIZE = (640, 480)
SAVE_DIR = "../data/imgs/"
os.makedirs(SAVE_DIR, exist_ok=True)
import matplotlib
matplotlib.use('TkAgg')

data = pd.DataFrame()


with open('../data/configs/inspection_info.json', 'r') as f:
    inspection_info = json.load(f)



# Count total iterations for progress bar
total = sum(inspection_info[f]["number_of_shots"] for f in inspection_info)

for filename, ith_shot in tqdm(
    ((f, s) for f in inspection_info for s in range(inspection_info[f]["number_of_shots"])),
    total=total
):
    # Read the US data (per shot)
    data_insp = file_m2k.read(
        DATA_ROOT + filename,
        freq_transd=5,
        bw_transd=0.5,
        tp_transd='gaussian',
        sel_shots=ith_shot
    )


    time_grid = data_insp.time_grid[:, 0]
    alpha_grid = np.arange(-45, 45 + .5, .5)


    # Converting channels to S-scan:
    channels_insp = data_insp.ascan_data[..., 0]
    sscan = REDUCTION_METHOD(channels_insp, axis=2)
    sscan_envelope = envelope(sscan, axis=0)
    sscan_max = sscan_envelope.max()

    sscan_log = np.log10(sscan_envelope / sscan_max + LOG_CTE)

    # Normalize to 0–65535 for 16-bit PNG
    img = 65535 * (sscan_log - sscan_log.min()) / (sscan_log.max() - sscan_log.min())
    img = img.astype(np.uint16)

    # Convert to PIL image (16-bit grayscale)
    im = Image.fromarray(img)

    # Get original size
    width, height = im.size

    match inspection_info[filename]["inspection type"]:
        case "reference no pipe":
            has_flaw = False
        case "static rbh":
            has_flaw = True
        case "active sweep sdh":
            has_flaw = True
        case "passive sweep rbh":
            has_flaw = False
            for interval in inspection_info[filename]["flaw_shot"]:
                if interval[0] <= ith_shot <= interval[1]:
                    has_flaw = True
        case _:
            raise ValueError


    # Enlarge horizontally ×5 with interpolation
    new_width = width * 5
    im_resized = im.resize((new_width, height), resample=Image.LANCZOS)  # or LANCZOS for higher quality

    # Save as 16-bit PNG
    # im = Image.fromarray(img)
    im_resized.save(os.path.join(SAVE_DIR, f"{filename}_shot{ith_shot:04}_flaw{int(has_flaw)}.png"))