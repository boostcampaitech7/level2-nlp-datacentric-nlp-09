import pandas as pd

def ensemble_and_save_csv(csv_path1, csv_path2, output_path, score_column="naturalness_score"):
    # 두 CSV 파일 로드
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    # 동일한 ID 또는 인덱스 기준으로 데이터 병합 (필요한 경우 조정)
    # 여기서는 인덱스를 기준으로 병합
    df1[score_column] = (df1[score_column] + df2[score_column]) / 2

    # 새로운 CSV 파일로 저장
    df1.to_csv(output_path, index=False)
    print(f"Ensembled CSV saved to {output_path}")

# 사용 예시

csv_path1 = "./data/preprocessed/yourdata.csv"
csv_path2 = "./data/preprocessed/yourdata.csv"
output_path = "yourdata.csv"
ensemble_and_save_csv(csv_path1, csv_path2, output_path)