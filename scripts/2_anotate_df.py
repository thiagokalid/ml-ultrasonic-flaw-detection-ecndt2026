import json
import pandas as pd
from utils import *
import re
import cv2
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_PATH = DATA_ROOT / "dataset"
MODELS_PATH = DATA_ROOT / "models"
US_DATA = DATA_ROOT / "us_dataset"
CONFIGS_PATH = DATA_ROOT / "configs"

DEFAULT_M2K_CONFIG = {
    "freq_transd":5,
    "bw_transd":0.5,
    "tp_transd":'gaussian'
}

TILES_PARAMS = {
    "overlapping": False,
    "num_tiles_xaxis": 20,
    "num_tiles_zaxis": 10,
}

if __name__ == "__main__":
    df = pd.read_pickle(DATASET_PATH / "dataset.pkl")

    # Hard-coded angle span based on used delay-law:
    alpha_grid = np.radians(np.arange(-45, 45 + .5, .5))

    # Read anotations:
    with open(CONFIGS_PATH / "_annotations.coco.json", "r") as f:
        annotations_insp = json.load(f)
    id_to_info = {img["id"]: img for img in annotations_insp["images"]}

    SSCAN_SIZE = (1875, 181)

    # Read data from these acquisitions:
    with open(CONFIGS_PATH / "inspection_info.json", "r") as f:
        insp_info = json.load(f)
    filename_list = insp_info.keys()
    filename_list = [list(filename_list)[0]]

    for ann in tqdm(annotations_insp["annotations"]):
        img_info = id_to_info[ann["image_id"]]
        filename = img_info["file_name"]
        m2k_filename = filename[:filename.find("m2k") - 1] + ".m2k"

        m2k_filename = re.sub(r"_38_5degree_", r"_38.5degree_", m2k_filename)

        shot = int(filename.split("_")[-3].replace("shot", ""))

        # Build mask
        mask = np.zeros(shape=SSCAN_SIZE, dtype=np.uint8)
        for seg in ann["segmentation"]:
            pts = np.array(seg).reshape(-1, 2).astype(int)
            pts[:, 0] = pts[:, 0] / 5
            cv2.fillPoly(mask, [pts], 1)

        selection = (df["filename"] == m2k_filename) & (df["shot"] == shot)
        df_selected = df[selection]

        if len(df_selected) > 0:
            file_info = insp_info[m2k_filename]
            t_outer, t_inner = file_info["surface_position"]
            data_insp = file_m2k.read(str(M2K_PATH / m2k_filename), read_ascan=False, *DEFAULT_M2K_CONFIG)
            time_grid = data_insp.time_grid[:, 0]

            new_flags = mark_tiles(df_selected, mask, time_grid, alpha_grid, t_outer, t_inner, TILES_PARAMS)
            df.loc[selection, "contain_flaw"] = df.loc[selection, "contain_flaw"].fillna(0) + new_flags

        else:
            pass

    df.loc[df["contain_flaw"] >= 1, "contain_flaw"] = 1
    df.loc[df["contain_flaw"] != 1, "contain_flaw"] = 0

    # Overwrite the dataset with the new annotated version:
    df.to_pickle(DATASET_PATH / "dataset.pkl")