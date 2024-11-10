import pandas as pd
import MeCab

# CSV 파일 경로
file_path = 'data/augmented/train_llm_v1.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path)

# " ' , * # 제거
data = data.applymap(lambda x: str(x).replace('"', '').replace("'", '').replace(',', '').replace('*', '').replace('#', '') if isinstance(x, str) else x)

# 'text' 컬럼에서 첫 번째 스페이스 이후의 문자열을 추출
data['text'] = data['text'].str.split(' ', n=1).str[1]

# 인덱스 추가 (0부터 시작하는 기본 인덱스 사용)
data.insert(0, 'ID', range(len(data)))

##################################################################

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

# 'text' 컬럼에서 명사만 추출하여 다시 'text' 컬럼에 저장
data['text'] = data['text'].apply(extract_nouns)

# 명사가 없는 행 필터링
data = data[data['text'].str.strip() != '']

# 결과를 CSV 파일로 저장
data.to_csv('data/augmented/train_llm_v2.csv', index=False, encoding='utf-8-sig')
