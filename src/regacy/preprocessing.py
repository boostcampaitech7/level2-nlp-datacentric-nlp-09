import pandas as pd
import re

def replace_special_chars_with_space(text):
    """ASCII 코드 특수문자를 스페이스로 대체하는 함수"""
    if not isinstance(text, str):
        return text
    # ASCII 코드 33번부터 126번에 해당하는 특수문자들을 스페이스로 대체
    return re.sub(r"[!-~]", " ", text)

def process_dataframe(df, text_column="text"):
    """데이터프레임의 text 컬럼에서 ASCII 특수문자를 스페이스로 대체"""
    df[text_column] = df[text_column].apply(replace_special_chars_with_space)
    return df

# 예시 실행
if __name__ == "__main__":
    # 파일 로드
    input_path = "./data/preprocessed/pipeline1_step.csv"  # 예시 파일 경로
    output_path = "./data/preprocessed/yourdata.csv"
    df = pd.read_csv(input_path)
    

    filtered_df = process_dataframe(df)
    
    # 파일 저장
    filtered_df.to_csv(output_path, index=False)
    print(f"saved to {output_path}")
