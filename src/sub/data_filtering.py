import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('data/preprocessed/train_noise_v2.csv')

# 영어 + 숫자 + 특수문자 비율을 계산하는 함수 (공백 미포함)
def non_korean_ratio(text):
    non_korean_count = sum(1 for char in text if (33 <= ord(char) <= 126))
    return non_korean_count / len(text) if len(text) > 0 else 0

# 영어 + 숫자 + 특수문자 비율이 50% 미만인 데이터를 필터링
df_under_50 = df[df['text'].apply(non_korean_ratio) < 0.5]

# 결과를 각각 새로운 CSV 파일로 저장
df_under_50.to_csv('data/preprocessed/train_noise_v4.csv', index=False)
