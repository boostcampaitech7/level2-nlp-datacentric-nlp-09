import pandas as pd

def replace_dots_with_space(df, text_column="text"):
    """특수문자 '...'을 공백으로 대체하는 함수"""
    df[text_column] = df[text_column].str.replace("...", " ", regex=False)
    df[text_column] = df[text_column].str.replace("…", " ", regex=False)
    return df

# 예시 실행
if __name__ == "__main__":
    # 파일 로드
    input_path = "./data/raw/train.csv"  # 예시 파일 경로
    output_path = "./data/preprocessed/preprocessed_train.csv"
    df = pd.read_csv(input_path)
    
    # '...'을 공백으로 대체
    df = replace_dots_with_space(df)
    
    # 파일 저장
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to {output_path}")
