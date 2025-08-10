import pandas as pd

PKL_PATH = "../../data/pkl/"

df1 = pd.read_pickle(PKL_PATH + 'acoustic_lens_sscan_dataset_A.pkl')
df2 = pd.read_pickle(PKL_PATH + 'acoustic_lens_sscan_dataset_B.pkl')
df3 = pd.read_pickle(PKL_PATH + 'acoustic_lens_sscan_dataset_C.pkl')

merged_df = pd.concat([df1, df2, df3])
merged_df.to_pickle(PKL_PATH + "acoustic_lens_sscan_dataset.pkl")