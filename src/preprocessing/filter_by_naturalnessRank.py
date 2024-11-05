import pandas as pd

def filter_by_naturalness_percentile(df, output_path, score_column="naturalness_score", min_percentile=0.1, max_percentile=1):
    """자연스러움 점수의 특정 분위수 범위에 해당하는 데이터를 필터링하여 저장하는 함수"""
    # 상하위 분위수 임계값 계산
    lower_threshold = df[score_column].quantile(min_percentile)
    upper_threshold = df[score_column].quantile(max_percentile)
    
    # 지정한 분위수 범위에 포함되는 데이터 필터링
    filtered_df = df[(df[score_column] >= lower_threshold) & (df[score_column] <= upper_threshold)]
    
    # 필터링된 데이터 저장
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered data with naturalness score between {min_percentile*100}% and {max_percentile*100}% saved to {output_path}")

# 예시 실행
if __name__ == "__main__":
    # 파일 로드
    input_path = "./data/preprocessed/with_naturalness_score_preprocessed_train_v3.csv"  # 입력 파일 경로
    output_path = "./data/preprocessed/preprocessed_train_v3.csv"
    df = pd.read_csv(input_path)
    
    # 상위 10% 미만 자연스러움 점수 데이터 필터링 및 저장
    filter_by_naturalness_percentile(df, output_path)
