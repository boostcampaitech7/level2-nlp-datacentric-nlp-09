import pandas as pd

# 데이터 로드
train_aug = pd.read_csv("data/augmented/train_aug_mb.csv")
train_clean = pd.read_csv("data/preprocessed/train_clean_v2.csv")

# train_aug의 ID와 train_clean의 ID 비교 후 target 값 수정
for idx, row in train_aug.iterrows():
    if row['ID'] in train_clean['ID'].values:
        # 동일한 ID를 가진 train_clean의 target 값을 train_aug의 target 값으로 업데이트
        train_clean.loc[train_clean['ID'] == row['ID'], 'target'] = row['target']

# 수정된 train_clean 데이터 저장
train_clean.to_csv("data/preprocessed/train_clean_v3.csv", index=False)
