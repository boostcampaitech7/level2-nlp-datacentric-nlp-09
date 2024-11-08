import pandas as pd
import re

def calculate_special_char_ratio(text):
    """텍스트에서 특수문자의 비율을 계산하는 함수"""
    if not isinstance(text, str):
        return 0
    # ASCII 특수문자 범위에 해당하는 문자들
    special_chars = re.findall(r"[!-/:-@[-`{-~]", text)
    # 특수문자 비율 계산
    return len(special_chars) / len(text) if len(text) > 0 else 0

def add_special_char_ratio(df, text_column="text"):
    """특수문자 비율을 계산하여 데이터프레임에 추가하는 함수"""
    df['special_char_ratio'] = df[text_column].apply(calculate_special_char_ratio)
    return df

# 예시 실행
if __name__ == "__main__":
    # 파일 로드
    input_path = "./data/preprocessed/preprocessed_train.csv"  # 예시 파일 경로
    output_path = "./data/preprocessed/with_special_char_ratio_train.csv"
    df = pd.read_csv(input_path)
    
    # 특수문자 비율 추가
    df = add_special_char_ratio(df)
    
    # 파일 저장
    df.to_csv(output_path, index=False)
    print(f"Processed file with special character ratio saved to {output_path}")
