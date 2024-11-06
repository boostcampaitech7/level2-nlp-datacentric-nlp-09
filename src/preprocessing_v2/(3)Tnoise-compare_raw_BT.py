import pandas as pd
import re

raw = pd.read_csv('data/preprocessed_v2/Tnoise_1073.csv')
bt = pd.read_csv('data/preprocessed_v2/Tnoise_BT_1073.csv')
result = raw.copy()

# korean ratio
raw['kr_ratio'] = raw['text'].apply(lambda x: len(re.findall('[가-힣]', x)) / len(x))
# sort

bt['kr_ratio'] = bt['text'].apply(
    lambda x: len(re.findall('[가-힣]', x)) / len(x) if isinstance(x, str) and len(x) > 0 else 0
)

# compare the ratio
for i in range(len(raw)):
    if raw.at[i, 'kr_ratio'] < bt.at[i, 'kr_ratio']:
        result.at[i, 'text'] = bt.at[i, 'text']
    else:
        result.at[i, 'text'] = raw.at[i, 'text']
result

result.to_csv('data/preprocessed_v2/Tnoise_comp_1073.csv', index=False)