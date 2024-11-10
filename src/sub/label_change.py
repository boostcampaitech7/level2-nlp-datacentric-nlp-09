import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data/preprocessed/train_normal.csv')

# target 컬럼 숫자를 문자로 변환하는 맵핑
mapping = {
    0: '정치',
    1: '경제',
    2: '사회',
    3: '생활문화',
    4: '세계',
    5: 'IT과학',
    6: '스포츠'
}

# target 컬럼 변환
df['target'] = df['target'].map(mapping)

# 변환된 데이터프레임 저장
df.to_csv('data/preprocessed/train_label_change.csv', index=False)
