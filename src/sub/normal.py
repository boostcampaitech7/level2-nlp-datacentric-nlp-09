import pandas as pd
import re

# CSV 파일 불러오기
df = pd.read_csv('data/raw/train.csv')

# 특수문자를 제외한 글자 수를 계산하는 함수
def special_char_ratio(text):
    # 전체 길이 중 특수문자가 차지하는 비율 계산
    special_chars = re.findall(r'[^\w\s]', text)  # 특수문자만 찾기
    return len(special_chars) / len(text) if len(text) > 0 else 0

# 특수문자 비율이 5% 이하인 행만 필터링
df_filtered = df[df['text'].apply(special_char_ratio) <= 0.05]

# 결과를 새로운 CSV 파일로 저장
df_filtered.to_csv('data/preprocessed/train_normal.csv', index=False)
