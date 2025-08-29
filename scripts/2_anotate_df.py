import json
import pandas as pd
from utils import *
import re
import cv2
from tqdm import tqdm

M2K_PATH = "../data/m2k/"
CONFIGS_PATH = "../data/configs/"
PKL_PATH = "../data/pkl/"
ANOTATION_PATH = "../data/imgs_anotated/"

DEFAULT_M2K_CONFIG = {
    "freq_transd":5,
    "bw_transd":0.5,
    "tp_transd":'gaussian'
}

SUBROI_PARAMS = {
    "overlapping": False,
    "num_subrois_xaxis": 20,
    "num_subrois_zaxis": 10,
}

if __name__ == "__main__":
    df = pd.read_pickle(PKL_PATH + "dataset.pkl")

    # Hard-coded angle span based on used delay-law:
    alpha_grid = np.radians(np.arange(-45, 45 + .5, .5))

    # Read anotations:
    with open(ANOTATION_PATH + f"/_annotations.coco.json", "r") as f:
        annotations_insp = json.load(f)
    id_to_info = {img["id"]: img for img in annotations_insp["images"]}

    SSCAN_SIZE = (1875, 181)

    # Read data from these acquisitions:
    with open(CONFIGS_PATH + f"/inspection_info.json", "r") as f:
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

            # Extract outer and inner surface temporal position:
            t_outer, t_inner = file_info["surface_position"]

            data_insp = file_m2k.read(M2K_PATH + m2k_filename, sel_shots=0, *DEFAULT_M2K_CONFIG)
            time_grid = data_insp.time_grid[:, 0]

            df.loc[selection, "contain_flaw"] = mark_subsscans(df_selected, mask, time_grid, alpha_grid, t_outer, t_inner,
                                                           SUBROI_PARAMS)
        else:
            pass

    df.loc[df["contain_flaw"] != 1, "contain_flaw"] = 0

    # Convert to dataframe:
    df.to_pickle(PKL_PATH + "dataset_annotated" + ".pkl")