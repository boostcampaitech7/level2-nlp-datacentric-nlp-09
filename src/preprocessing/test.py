import pandas as pd

csv1_path = "./data/preprocessed/unique_ids_predict.csv"  # 예시 파일 경로
csv2_path = "./data/preprocessed/preprocessed_train_v4.csv"

# 두 CSV 파일 읽기
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# # ID 열에서 겹치지 않는 행만 선택 (양쪽 데이터프레임에 없는 ID)
# unique_df1 = df1[~df1['ID'].isin(df2['ID'])]
# unique_df2 = df2[~df2['ID'].isin(df1['ID'])]

# # 겹치지 않는 ID 행들을 하나의 데이터프레임으로 합치기
# unique_rows = pd.concat([unique_df1, unique_df2])

# # 결과를 새로운 CSV 파일로 저장
# unique_rows.to_csv('unique_ids.csv', index=False)

# print("겹치지 않는 ID만 저장된 CSV 파일이 생성되었습니다.")

# 두 데이터프레임을 행 방향으로 결합
combined_df = pd.concat([df1, df2], ignore_index=True)

# 결과를 새로운 CSV 파일로 저장
combined_df.to_csv('combined_rows.csv', index=False)

print("두 CSV 파일이 행으로 합쳐진 CSV 파일이 생성되었습니다.")