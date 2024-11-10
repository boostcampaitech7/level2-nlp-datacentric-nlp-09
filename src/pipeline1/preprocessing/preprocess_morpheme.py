import pandas as pd
from konlpy.tag import Kkma

"""
Extracts and filters specific morphemes (parts of speech) from text data in a CSV file for topic classification, 
then saves the modified text data to a new CSV file.

This script performs the following tasks:
1. Loads text data from a CSV file.
2. Initializes the Kkma morphological analyzer.
3. Identifies and extracts morphemes with tags relevant to topic classification, specifically nouns (default is general and proper nouns).
4. Replaces the original text in the 'text' column with only the extracted morphemes.
5. Saves the modified data to a new CSV file.

Attributes:
    csv_path (str): Path to the input CSV file containing text data.
    important_tags (list): List of part-of-speech tags deemed important for topic classification.
                           Default tags are general nouns ('NNG') and proper nouns ('NNP').

Functions:
    extract_important_morphemes(text):
        Extracts and returns morphemes with tags specified in `important_tags`.
        Args:
            text (str): The input text to process.
        Returns:
            str: A string containing only the morphemes with specified tags.

Usage:
    Run this script to preprocess text data in a CSV file, focusing on morphemes relevant to topic classification.
    The processed data is saved to a new CSV file.

Example:
    ```
    python extract_important_morphemes.py
    ```
    This will save the processed data to a new CSV file at the specified output path.

Returns:
    None
"""


# Mecab 형태소 분석기 초기화
mecab = Kkma()

csv_path = "./data/preprocess/pipeline1_combined_rows.csv"
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
df.to_csv('./data/preprocess/pipeline1_combined_rows_morpheme.csv', index=False)

print("주제 분류에 중요한 형태소가 추출된 텍스트가 저장된 CSV 파일이 생성되었습니다.")