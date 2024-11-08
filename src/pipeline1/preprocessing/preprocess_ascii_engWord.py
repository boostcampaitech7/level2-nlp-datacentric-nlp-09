import pandas as pd
from collections import Counter
import re

# CSV 파일 경로
input_csv_path = "./data/preprocessed/pipeline1_step3.csv"
output_csv_path = "./data/preprocessed/pipeline1_step4.csv"

# CSV 파일 로드
df = pd.read_csv(input_csv_path)

# 영어 단어 빈도를 계산하고 필터링하는 함수
def get_filtered_words(df, min_frequency=2):
    """텍스트에서 빈도 2 이상인 대문자 2글자 이상의 영어 단어를 필터링"""
    word_counts = {}

    # 각 텍스트에서 영어 단어 추출 및 빈도 계산
    for text in df['text']:
        words = re.findall(r'\b[A-Z]{2,}\b', str(text))  # 대문자 2글자 이상의 단어만 추출
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    # 빈도가 min_frequency 이상인 단어만 필터링하여 반환
    filtered_words = {word for word, count in word_counts.items() if count >= min_frequency}
    return filtered_words

# 필터링된 단어 집합 생성
filtered_words = get_filtered_words(df)

# 특수문자를 대체하는 함수
def replace_special_chars_conditionally(text):
    """조건에 따라 특수문자를 스페이스로 대체하는 함수"""
    if not isinstance(text, str):
        return text

    # 필터링된 단어는 그대로 두고 나머지 특수문자만 스페이스로 대체
    # 패턴: 필터링된 단어는 제외하고, ASCII 33번~126번의 특수문자를 스페이스로 대체
    pattern = r'\b(' + '|'.join(re.escape(word) for word in filtered_words) + r')\b|[!-~]'
    
    # 조건을 만족하는 단어는 그대로 유지하고, 나머지 특수문자는 스페이스로 대체
    return re.sub(pattern, lambda x: x.group(0) if x.group(1) else ' ', text)

# 데이터프레임에 함수 적용
df['text'] = df['text'].apply(replace_special_chars_conditionally)

# 결과 저장
df.to_csv(output_csv_path, index=False)
print(f"Processed data saved to {output_csv_path}")


