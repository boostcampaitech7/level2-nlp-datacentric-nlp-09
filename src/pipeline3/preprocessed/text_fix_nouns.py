import pandas as pd
import os

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data/preprocessed')
data1 = pd.read_csv(os.path.join(DATA_DIR, 'label_fix.csv'))

# 특수기호가 있는 행의 'text' 컬럼에 'nouns' 컬럼 값을 복사
data1['text'] = data1.apply(lambda row: row['nouns'] if any(not char.isalnum() and not char.isspace() for char in row['text']) else row['text'], axis=1)

data1.to_csv("data/preprocessed/text_fix_nouns.csv", index=False, encoding='utf-8-sig')

