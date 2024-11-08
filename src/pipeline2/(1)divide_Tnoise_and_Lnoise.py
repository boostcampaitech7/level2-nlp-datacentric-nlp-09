import pandas as pd
import re
import os

"""
이 코드는 주어진 raw train 데이터를 레이블 노이즈(Label Noise)와 텍스트 노이즈(Text Noise)가 포함된 두 개의 파일로 분할하는 코드입니다.
각각의 노이즈는 앞으로 Pipeline에서 독립적으로 처리됩니다.

코드 설명:
1. 'train.csv' 파일을 불러오고, 복사본을 생성하여 처리합니다.
2. 텍스트에서 한글 및 공백을 제외한 다른 모든 문자를 '*'로 대체합니다.
3. 텍스트 내 '*'의 비율(noise ratio)을 계산하고, 이를 기준으로 정렬하여 상위 1200개의 레코드를 레이블 노이즈로 분류합니다.
4. 나머지 데이터 중 noise ratio가 0.5 미만인 데이터를 텍스트 노이즈로 분류합니다.
5. 각 분류에 해당하는 데이터를 파일로 저장합니다:
   - Label Noise 데이터는 'Lnoise_1200.csv'에 저장.
   - Text Noise 데이터는 'Tnoise_1073.csv'에 저장.
"""

# make directory if not exists
os.makedirs('data/preprocessed', exist_ok=True)

# load raw data
raw = pd.read_csv('data/raw/train.csv')

# replace non-korean characters with '*'
df = raw.copy()
sub = '*'
df['text'] = df['text'].apply(lambda x: re.sub(r'[^가-힣\s]', sub, x))

# calculate noise ratio
df['noise_ratio'] = df['text'].str.count('\*') / df['text'].str.len()

# sort by noise ratio
df.sort_values(by='noise_ratio', ascending=True, inplace=True)

# split data into Label Noise and Text Noise
Lnoise = df.iloc[:1200].copy()
Tnoise = df.iloc[1200:].copy()
Tnoise = Tnoise[Tnoise['noise_ratio'] < 0.5]

# save data
Lnoise_id = Lnoise['ID'].to_list()
Tnoise_id = Tnoise['ID'].to_list()
Lnoise_data = raw[raw['ID'].isin(Lnoise_id)].copy()
Tnoise_data = raw[raw['ID'].isin(Tnoise_id)].copy()
Tnoise_data.to_csv('data/preprocessed/Tnoise_1073.csv', index=False)
Lnoise_data.to_csv('data/preprocessed/Lnoise_1200.csv', index=False)