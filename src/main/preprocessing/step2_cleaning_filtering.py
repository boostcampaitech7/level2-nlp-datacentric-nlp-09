import pandas as pd
import MeCab

# Mecab 형태소 분석기 생성
mecab = MeCab.Tagger()

# 명사 추출 후 띄어쓰기로 이어 붙이는 함수 정의
def extract_nouns(text):
    parsed = mecab.parse(text)  # 텍스트 분석
    nouns = []
    for line in parsed.splitlines()[:-1]:  # 마지막 빈 줄 제외
        word, feature = line.split('\t')
        if 'NN' in feature:  # 명사인 경우
            nouns.append(word)
    return ' '.join(nouns)  # 명사 리스트를 띄어쓰기로 연결

# CSV 파일 불러오기
noise_df = pd.read_csv('data/preprocessed/train_noise_v2.csv')
clean_df = pd.read_csv('data/preprocessed/train_clean_v2.csv')

# 'text' 컬럼에서 명사만 추출하여 다시 'text' 컬럼에 저장
noise_df['text'] = noise_df['text'].apply(extract_nouns)
clean_df['text'] = clean_df['text'].apply(extract_nouns)

# 명사가 없는 행 필터링
noise_df = noise_df[noise_df['text'].str.strip() != '']
clean_df = clean_df[clean_df['text'].str.strip() != '']

# 결과를 CSV 파일로 저장
noise_df.to_csv('data/preprocessed/train_noise_v3.csv', index=False, encoding='utf-8-sig')
clean_df.to_csv('data/preprocessed/train_clean_v2_mb.csv', index=False, encoding='utf-8-sig')
