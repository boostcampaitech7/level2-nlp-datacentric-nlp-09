import pandas as pd

raw = pd.read_csv('data/raw/train.csv')
df = pd.read_csv('data/preprocessed/noise2asterisk.csv')

# '*'/len ratio in text columns
df['noise_ratio'] = df['text'].str.count('\*') / df['text'].str.len()

# sort by noise ratio
df.sort_values(by='noise_ratio', ascending=True, inplace=True)
Lnoise = df.iloc[:1200].copy()
Tnoise = df.iloc[1200:].copy()
Tnoise = Tnoise[Tnoise['noise_ratio'] < 0.5]

Lnoise_id = Lnoise['ID'].to_list()
Tnoise_id = Tnoise['ID'].to_list()

Lnoise_data = raw[raw['ID'].isin(Lnoise_id)].copy()
Tnoise_id_data = raw[raw['ID'].isin(Tnoise_id)].copy()


Lnoise_data.to_csv('data/preprocessed/asterisk2GTandLnoise.csv', index=False)
Tnoise_id_data.to_csv('data/preprocessed/asterisk2Tnoise_HighRatioCut.csv', index=False)