import json
import pandas as pd
from datetime import datetime
from utils import *

M2K_PATH = "../data/m2k/"
CONFIGS_PATH = "../data/configs/"
PKL_PATH = "../data/pkl/"
ANOTATION_PATH = "../data/imgs_anotated/"

PROCESSING_PARAMS = {
    "subtraction": False,
    "reduction_method": "median",
    "scale": "log",
}

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
    # Read data from these acquisitions:
    with open(CONFIGS_PATH + f"/inspection_info.json", "r") as f:
        insp_info = json.load(f)
    filename_list = insp_info.keys()

    # Data-structure to be converted into dataframe later:
    DATAFRAME_HEADER = ["filename", "delta_t", "delta_ang", "shot", "subroi_idx", "t_outer", "t_inner", "sscan_max", "contain_flaw",
                        "subroi_limits", "sub_sscan"]
    df_rows = {key_: [] for key_ in DATAFRAME_HEADER}

    # Hard-coded angle span based on used delay-law:
    alpha_grid = np.radians(np.arange(-45, 45 + .5, .5))

    # Useful information:
    total_iter = np.sum([insp_info[filename]["number_of_shots"] for filename in filename_list])
    number_of_subrois = (SUBROI_PARAMS["num_subrois_xaxis"] - 1) * (SUBROI_PARAMS["num_subrois_zaxis"] - 1)

    # Print start time
    start_time = datetime.now()
    print(f"Computation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Iterate over all US acquisiton files and read it:
    progress = 0
    print(f"Progress: {progress}/{total_iter} ({progress / total_iter:.1%})", end="\r")
    for filename in filename_list:
        file_info = insp_info[filename]
        number_of_shots = file_info["number_of_shots"]

        for shot in range(number_of_shots):
            data_insp = file_m2k.read(M2K_PATH + filename, sel_shots=shot, *DEFAULT_M2K_CONFIG)
            time_grid = data_insp.time_grid[:, 0]

            reduction_method = get_reduction_method(PROCESSING_PARAMS["reduction_method"])

            # Get the reference inspection for that given file:
            channels_ref = get_ref(data_insp.ascan_data[..., 0].shape, M2K_PATH + file_info["ref_filename"], PROCESSING_PARAMS["subtraction"], DEFAULT_M2K_CONFIG)

            # Extract outer and inner surface temporal position:
            t_outer, t_inner = file_info["surface_position"]


            # Apply (or not) subtraction:
            channels_insp = data_insp.ascan_data[..., 0]
            channels = channels_insp - channels_ref if PROCESSING_PARAMS["subtraction"] else channels_insp
            sscan = reduction_method(channels, axis=-1)

            # Apply (or not) image enhancement techniques:
            sscan = apply_preprocessing(sscan, envelope=True, normalize=True)

            # Apply (or not) different scale:
            sscan = apply_scale(sscan, PROCESSING_PARAMS["scale"])

            # Split the sscan into differente regions of interest and append to df rows:
            split_sscan2subrois(
                df_rows, sscan, shot, time_grid, alpha_grid, t_outer, t_inner, SUBROI_PARAMS)
            df_rows["filename"].extend(number_of_subrois * [filename])

            # Update progress counter
            progress += 1
            elapsed = datetime.now() - start_time
            elapsed_str = str(elapsed).split(".")[0]
            print(
                f"Progress: {progress}/{total_iter} ({progress / total_iter:.1%}) "
                f"- elapsed {elapsed_str}",
                end="\r"
            )

    # Convert to dataframe:
    df = pd.DataFrame(df_rows)
    df.to_pickle(PKL_PATH + "dataset" + ".pkl")