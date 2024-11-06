import pandas as pd
import re

df = pd.read_csv('data/raw/train.csv')

# 한글, 공백을 제외한 모든 문자를 *로 대체
# 노이즈가 아닌 숫자, 영문자도 대체됨
sub = '*'
df['text'] = df['text'].apply(lambda x: re.sub(r'[^가-힣\s]', sub, x))

# 저장
df.to_csv(f'data/preprocessed/noise2asterisk.csv', index=False)