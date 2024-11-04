import streamlit as st
import pandas as pd
import os
import importlib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing import preprocessing  # 전처리 모듈을 src 폴더에서 임포트

# 전처리 모듈 강제 재로딩 (코드 수정 시 반영)
importlib.reload(preprocessing)

# 원본 CSV 파일이 있는 디렉토리 경로
raw_data_path = './data/raw'
preprocessed_data_path = './data/preprocessed'

# 원본 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
selected_file = st.selectbox("Choose a CSV file", csv_files)

# 선택한 CSV 파일 경로
csv_file_path = os.path.join(raw_data_path, selected_file)

try:
    # 원본 데이터 로드
    df = preprocessing.load_data(csv_file_path)

    # 전처리 수행
    filtered_df = preprocessing.preprocess_data(df)

    # 임계값 표시
    TEXT_LENGTH_THRESHOLD, TOKEN_LENGTH_THRESHOLD = preprocessing.get_thresholds()
    st.subheader("Threshold Values")
    st.write(f"Text Length Threshold: {TEXT_LENGTH_THRESHOLD}")
    st.write(f"Token Count Threshold: {TOKEN_LENGTH_THRESHOLD}")

    # 필터링 결과 시각화
    st.subheader("Filtering Results")
    st.write("Original Data Size:", len(df))
    st.write("Filtered Data Size:", len(filtered_df))
    st.write("Excluded Data Count:", len(df) - len(filtered_df))

    # 전처리된 데이터를 data/preprocessed 폴더에 저장
    preprocessed_file_path = os.path.join(preprocessed_data_path, f"preprocessed_{selected_file}")
    filtered_df.to_csv(preprocessed_file_path, index=False)
    st.write(f"Preprocessed data saved to {preprocessed_file_path}")

    # 전처리된 데이터 미리보기
    st.subheader("Filtered Data Preview")
    st.write(filtered_df.head())

except FileNotFoundError:
    st.error(f"File not found at path: {csv_file_path}. Please check the path.")
