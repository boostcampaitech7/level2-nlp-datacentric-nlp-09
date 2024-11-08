import pandas as pd

recovered_Tnoise_only = pd.read_csv('data/preprocessed/recovered_Tnoise_only.csv')
recovered2relabered_Lnoise_only = pd.read_csv('data/preprocessed/recovered2relabeled_Lnoise_only.csv')

# concatenate the two dataframes
recovered_all_data = pd.concat([recovered_Tnoise_only, recovered2relabered_Lnoise_only], ignore_index=True)

# save the concatenated dataframe
recovered_all_data.to_csv('data/preprocessed/concat_recover_relabel.csv', index=False)