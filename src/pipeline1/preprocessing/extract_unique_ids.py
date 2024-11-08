import pandas as pd

train_path = "./data/raw/train.csv"  # 예시 파일 경로
csv_path = "./data/preprocessed/train_3dot_specialDel_v2.csv"

# 두 CSV 파일 읽기
df1 = pd.read_csv(train_path)
df2 = pd.read_csv(csv_path)

# ID 열에서 겹치지 않는 행만 선택 (양쪽 데이터프레임에 없는 ID)
unique_df1 = df1[~df1['ID'].isin(df2['ID'])]
unique_df2 = df2[~df2['ID'].isin(df1['ID'])]

# 겹치지 않는 ID 행들을 하나의 데이터프레임으로 합치기
unique_rows = pd.concat([unique_df1, unique_df2])

# 결과를 새로운 CSV 파일로 저장
unique_rows.to_csv('./data/preprocessed/unique_ids.csv', index=False)

print("겹치지 않는 ID만 저장된 CSV 파일이 생성되었습니다.")