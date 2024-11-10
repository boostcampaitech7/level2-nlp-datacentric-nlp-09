import pandas as pd
from collections import Counter
import re

"""
Filters and processes text data in a CSV file based on specified word frequency and special character conditions, 
then saves the processed data to a new CSV file.

This script performs the following tasks:
1. Loads text data from a CSV file.
2. Identifies English words in uppercase with two or more letters and counts their frequency across the text data.
3. Filters words that appear with a frequency of `min_frequency` or more (default is 2).
4. Replaces special characters in the text with spaces, while preserving the filtered high-frequency words.
5. Saves the processed text data to a new CSV file.

Attributes:
    input_csv_path (str): Path to the input CSV file containing text data.
    output_csv_path (str): Path to the output CSV file where processed results are saved.

Functions:
    get_filtered_words(df, min_frequency=2):
        Extracts and filters uppercase English words with a minimum frequency of `min_frequency`.
        Args:
            df (pd.DataFrame): The input DataFrame containing text data.
            min_frequency (int): Minimum frequency for a word to be preserved.
        Returns:
            set: A set of words with frequency greater than or equal to `min_frequency`.

    replace_special_chars_conditionally(text):
        Replaces special characters in the text with spaces, preserving words in `filtered_words`.
        Args:
            text (str): A string of text to be processed.
        Returns:
            str: Processed text with special characters replaced by spaces, preserving high-frequency words.

Usage:
    Run this script to process text in a CSV file, where:
    - High-frequency uppercase words with two or more letters are preserved.
    - Special characters are replaced with spaces except in the preserved words.
    The processed data is saved to `output_csv_path`.

Example:
    ```
    python filter_and_replace_special_chars.py
    ```
    This will save the processed data in `output_csv_path`.

Returns:
    None
"""



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


