import pandas as pd
import re

def replace_special_chars(text):
    """특수문자를 공백으로 대체하는 함수"""
    if not isinstance(text, str):
        return text
    return re.sub(r"[^\w\s]", " ", text)

def add_cleaned_text_as_new_rows(input_path, output_path, text_column="text"):
    """특수문자를 공백으로 대체한 텍스트를 새로운 행으로 추가한 CSV 파일 저장"""
    # 파일 로드
    df = pd.read_csv(input_path)

    # 특수문자를 공백으로 대체한 텍스트를 새 행으로 추가
    cleaned_texts = df[text_column].apply(replace_special_chars)
    cleaned_df = df.copy()
    cleaned_df[text_column] = cleaned_texts

    # 기존 데이터와 새로운 행을 결합
    augmented_df = pd.concat([df, cleaned_df], ignore_index=True)

    # 파일 저장
    augmented_df.to_csv(output_path, index=False)
    print(f"CSV with cleaned text added as new rows saved to {output_path}")

# 예시 실행
if __name__ == "__main__":
    input_path = "./data/preprocessed/preprocessed_train_v3_2.csv"  # 예시 파일 경로
    output_path = "./data/augmented/preprocessed_train_v4_2.csv"
    add_cleaned_text_as_new_rows(input_path, output_path)
