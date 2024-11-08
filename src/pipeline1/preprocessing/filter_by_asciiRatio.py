import pandas as pd
import re

def calculate_special_char_ratio(text):
    """텍스트에서 아스키코드 33~126번에 해당하는 특수문자 비율을 계산하는 함수"""
    if not isinstance(text, str):
        return 0
    # ASCII 코드 33번부터 126번까지의 특수문자 범위에 해당하는 문자들
    special_chars = re.findall(r"[!-~]", text)
    # 특수문자 비율 계산
    return len(special_chars) / len(text) if len(text) > 0 else 0

def filter_by_special_char_ratio(df, text_column="text", min_ratio=0, max_ratio=0.5):
    """특수문자 비율이 특정 범위에 있는 행만 필터링하는 함수"""
    filtered_df = df[df[text_column].apply(lambda x: min_ratio <= calculate_special_char_ratio(x) <= max_ratio)]
    return filtered_df


# 파일 로드
input_path = "./data/preprocessed/pipeline1_step2.csv"  # 예시 파일 경로
output_path = "./data/preprocessed/pipeline1_step3.csv"
df = pd.read_csv(input_path)

# 20% ~ 80% 범위의 데이터 필터링
filtered_df = filter_by_special_char_ratio(df)

# 파일 저장
filtered_df.to_csv(output_path, index=False)
print(f"Filtered file with special character ratio between 0% and 50% saved to {output_path}")
