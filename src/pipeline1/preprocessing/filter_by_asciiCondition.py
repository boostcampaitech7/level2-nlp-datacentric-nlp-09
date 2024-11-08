import pandas as pd
import re

# 특수문자 필터 설정
special_char_input = "]^[<>@'+/;!=\#`)~$*}{|:&_?,(-\""  # 찾고자 하는 특수문자 목록
input_csv_path = "./data/preprocessed/pipeline1_step1.csv"  # 원본 CSV 파일 경로
output_csv_path = "./data/preprocessed/pipeline1_step2.csv"  # 필터링된 결과를 저장할 CSV 파일 경로

# CSV 파일 로드
df = pd.read_csv(input_csv_path)

# 특수문자 조건 생성
special_chars = f"[{re.escape(special_char_input)}]"

# 추가 조건 정의
# '%'가 있을 때 앞에 숫자가 오지 않는 경우
percent_condition = r"(?<!\d)%"

# '.'가 있을 때:
# 1) '.' 앞에 '다'가 아닌 경우
# 2) '.'의 앞뒤에 숫자가 아닌 경우
dot_condition = r"(?<!다)\.|(?<!\d)\.(?!\d)"

# 정규 표현식을 통해 필터링 조건 생성
filter_pattern = f"{special_chars}|{percent_condition}|{dot_condition}"

# 조건에 맞는 행을 필터링
filtered_df = df[df['text'].str.contains(filter_pattern, regex=True, na=False)]

# 필터링된 데이터 저장
filtered_df.to_csv(output_csv_path, index=False)
print(f"Filtered data saved to {output_csv_path}")
