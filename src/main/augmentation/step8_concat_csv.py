import pandas as pd

# CSV 파일 읽기
df1 = pd.read_csv('data/preprocessed/train_noise_v3.csv')  # 첫 번째 CSV 파일
df2 = pd.read_csv('data/preprocessed/train_clean_v3.csv')  # 두 번째 CSV 파일
df3 = pd.read_csv('data/augmented/train_aug_mb.csv')  # 세 번째 CSV 파일
df4 = pd.read_csv('data/augmented/train_llm_v3.csv')  # 네 번째 CSV 파일

# 전체 데이터프레임 합치기
concat_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 연결된 데이터프레임을 새로운 CSV 파일로 저장
concat_df.to_csv('data/final/train_final.csv', index=False)
