import pandas as pd
from konlpy.tag import Kkma

# Mecab 형태소 분석기 초기화
mecab = Kkma()

csv_path = "./data/augmented/yourdata.csv"
# 주제 분류에 중요한 형태소 태그 설정 (명사, 동사, 형용사)
important_tags = ['NNG', 'NNP']#, 'VV', 'VA']

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 'text' 열에서 주제 분류에 중요한 형태소만 추출하여 대체
def extract_important_morphemes(text):
    morphs = mecab.pos(text)
    important_morphs = [word for word, pos in morphs if pos in important_tags]
    return ' '.join(important_morphs)

# 형태소 추출하여 'text' 열 대체
df['text'] = df['text'].apply(extract_important_morphemes)

# 결과를 새로운 CSV 파일로 저장
df.to_csv('yourdata.csv', index=False)

print("주제 분류에 중요한 형태소가 추출된 텍스트가 저장된 CSV 파일이 생성되었습니다.")