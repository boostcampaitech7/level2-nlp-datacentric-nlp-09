#행 결합

import pandas as pd

# train_path = "./data/raw/train.csv"  # 예시 파일 경로
csv1_path = "./data/augmented/combined_rows_aug_v3.csv"  # 예시 파일 경로
csv2_path = "./data/augmented/unique_ids_engWord_output_onlybt.csv"

# 두 CSV 파일 읽기
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# 두 데이터프레임을 행 방향으로 결합
combined_df = pd.concat([df1, df2], ignore_index=True)

# 결과를 새로운 CSV 파일로 저장
combined_df.to_csv('combined_rows_aug_v4.csv', index=False)

print("두 CSV 파일이 행으로 합쳐진 CSV 파일이 생성되었습니다.")