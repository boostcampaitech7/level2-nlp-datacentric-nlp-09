import pandas as pd
import re


raw = pd.read_csv('data/raw/train.csv')

df = raw.copy()
sub = '*'
df['text'] = df['text'].apply(lambda x: re.sub(r'[^가-힣\s]', sub, x))


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


Tnoise_id_data.to_csv('data/preprocessed_v2/Tnoise_1073.csv', index=False)
Lnoise_data.to_csv('data/preprocessed_v2/Lnoise_1200.csv', index=False)