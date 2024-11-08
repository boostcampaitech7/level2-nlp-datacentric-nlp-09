import pandas as pd

df = pd.read_csv('data/preprocessed_v2/Lnoise_Augmentation_4981_aya-Lrecovered.csv')
print(df.shape)

df = df.assign(topic=df['topic'].str.split('\n')).explode('topic')

df = df[df['topic'].notna()]
df = df[df['topic'] != '']
df = df[df['topic'] != 'None']

df = df[df['topic'].str.contains('topic', case=False)]

mapping = {
    '생활': 0,
    '문화': 0,
    '연예': 0,
    '예술': 0,
    '날씨': 0,
    '엔터테인먼트': 0,
    '스포츠': 1,
    '게임': 1,
    '정치': 2,
    '사회': 3,
    '환경': 3,
    '과학': 4,
    'it': 4,
    '기술': 4,
    '경제': 5,
    '세계': 6
}

# mapping의 key가 string에 포함되면 value로 변경
df['target'] = df.apply(
    lambda row: next((mapping[key] for key in mapping if key in str(row['topic']).lower()), row['target']),
    axis=1
)

df['topic'] = df['topic'].str.replace(' ', '', regex=False)  # 공백 제거
df['topic'] = df['topic'].str.extract(r':(.+)$')  # ':' 오른쪽의 내용 추출

df = df[df['target'] != -1]

df2 = pd.read_csv('data/preprocessed_v2/Tnoise_comp_recover_1073.csv')

# concat
df = pd.concat([df, df2], axis=0)

df.to_csv('data/preprocessed_v2/clean_aya.csv', index=False)


print(df.shape)
print(df['target'].value_counts())