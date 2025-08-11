import pandas as pd

PKL_PATH = "../../data/pkl/"

df_list = []
df_list.append(pd.read_pickle(PKL_PATH + 'acoustic_lens_sscan_dataset_A.pkl'))
df_list.append(pd.read_pickle(PKL_PATH + 'acoustic_lens_sscan_dataset_B.pkl'))
# df_list.append(pd.read_pickle(PKL_PATH + 'acoustic_lens_sscan_dataset_C.pkl'))

merged_df = pd.concat(df_list)
merged_df.to_pickle(PKL_PATH + "acoustic_lens_sscan_dataset.pkl")