import pandas as pd
from collections import Counter
import re

"""
이 코드는 LLM모델로 생성된 뉴스 헤드라인 데이터셋을 후처리하여 실제 헤드라인 데이터만 뽑는 작업을 수행하는 코드입니다.

코드 설명:
1. 헤드라인이 포함되어 있는 LLM 응답이 'text' 열에 들어있는 CSV 파일(`augmented_no_label.csv`)을 불러옵니다.
2. 'text' 열의 데이터를 개행 문자('\n') 기준으로 분리하여 새로운 행으로 확장합니다.
3. 빈 문자열과 텍스트 길이가 15자 미만 또는 40자 이상인 행을 제거합니다.
4. 특정 단어(예: '주제', '답변', '과제' 등)가 포함된 행을 제거합니다.
5. 텍스트의 한국어 비율을 계산하고, 0.4 이하의 비율을 가진 행을 제거합니다.
6. 'answer', 'Answer', '답:' 등의 문자열을 제거하고, 허용된 문자 외의 다른 문자를 제거하여 텍스트를 깨끗하게 만듭니다.
7. 최종 데이터프레임을 CSV 파일(`augmented_no_label-cleaned.csv`)로 저장합니다.
"""


df = pd.read_csv('data/augmented/augmented_no_label.csv')

# 1행 제거
df = df.drop(0)

# 'text' 열의 데이터를 '\n' 기준으로 분리하여 새로운 행으로 확장
df = df.set_index(['ID', 'target']).text.str.split('\n', expand=True).stack().reset_index(level=2, drop=True).reset_index()
df.columns = ['ID', 'target', 'text']
df = df[['ID', 'text', 'target']]

print(f'개행 기준 분리 행 수: {len(df)}')

# empty string 제거
df = df[df['text'] != '']
print(f'Empty string 제거 후 행 수: {len(df)}')

# 텍스트 길이 기반 제거
df['text_len'] = df['text'].str.len()
df = df[df['text_len'] >= 15]
df = df[df['text_len'] < 40]
print(f'텍스트 길이 기반 제거 후 행 수: {len(df)}')

remove_words = ['주제', '글자', '제목', '독자', '어조', '답변', '과제', '아이디어', '여러분', '설명', '당신', '?', '!']

# 단어들이 포함된 행 제거
df = df[~df['text'].apply(lambda x: any(word in x for word in remove_words))]
print(f'removed_words 단어들이 포함된 행 제거 후 행 수: {len(df)}')

# 새로운 열에 한국어 비율 추가
def korean_ratio(text):
    korean_count = len(re.findall(r'[가-힣]', text))
    total_count = len(text)
    if total_count == 0:
        return 0
    return korean_count / total_count

df['korean_ratio'] = df['text'].apply(korean_ratio)
df = df[df['korean_ratio'] > 0.4]
print(f'한국어 비율 추가 후 행 수: {len(df)}')

# 'text' 열의 데이터에서 'answer' 또는 'Answer' 또는 '답:' 문자열 제거
df['text'] = df['text'].apply(lambda x: re.sub(r'\b(answer|Answer|답:)\b', '', x).strip())

df['text'] = df['text'].apply(lambda x: ''.join([char if re.match(r'[가-힣a-zA-Z0-9\s,.\'\"]', char) else '' for char in x]))
df = df.drop(columns=['text_len', 'korean_ratio'])
print(f'최종 행 수: {len(df)}')
df.to_csv('data/augmented/aya-32b-cleaned.csv', index=False)