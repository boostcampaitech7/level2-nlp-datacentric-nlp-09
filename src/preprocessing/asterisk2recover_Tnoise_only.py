import pandas as pd

df = pd.read_csv('data/preprocessed/asterisk2Tnoise.csv')
print(df.shape)

df2 = pd.read_csv('data/preprocessed/asterisk2recover_Tnoise.csv')
print(df2.shape)

df_id_list = df['ID'].tolist()
df2_id_list = df2['ID'].tolist()

df2 = df2[df2['ID'].isin(df_id_list)]
print(f'After removing duplicates: {df2.shape}')

df2.to_csv('data/preprocessed/asterisk2recover_Tnoise_only.csv', index=False)