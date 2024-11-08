import pandas as pd
import re

"""
이 코드는 Back Translation을 통해 생성된 텍스트와 Text noise 텍스트를 비교하여 한글 비율이 더 높은 텍스트를 선택하여 저장하는 코드입니다.

코드 설명:
1. Text noise 데이터(`Tnoise_1073.csv`)와 Back Translation된 데이터(`Tnoise_BT_1073.csv`)를 불러옵니다.
2. 각 텍스트의 한글 비율을 계산하여 `kr_ratio` 컬럼에 저장합니다.
3. Text noise 텍스트와 번역된 텍스트를 비교하여 한글 비율이 더 높은 텍스트를 선택하고, 이를 결과 데이터프레임에 저장합니다.
4. 최종 결과는 `Tnoise_comp_1073.csv`로 저장됩니다.
"""

# load data
raw = pd.read_csv('data/preprocessed/Tnoise_1073.csv')
bt = pd.read_csv('data/preprocessed/Tnoise_BT_1073.csv')
result = raw.copy()

# calculate korean ratio
raw['kr_ratio'] = raw['text'].apply(lambda x: len(re.findall('[가-힣]', x)) / len(x))
bt['kr_ratio'] = bt['text'].apply(lambda x: len(re.findall('[가-힣]', x)) / len(x) if isinstance(x, str) and len(x) > 0 else 0)

# compare and select
for i in range(len(raw)):
    if raw.at[i, 'kr_ratio'] < bt.at[i, 'kr_ratio']:
        result.at[i, 'text'] = bt.at[i, 'text']
    else:
        result.at[i, 'text'] = raw.at[i, 'text']

# save data
result.to_csv('data/preprocessed/Tnoise_comp_1073.csv', index=False)