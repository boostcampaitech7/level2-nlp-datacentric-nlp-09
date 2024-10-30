import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import os
import random

# CSV 파일이 있는 디렉토리 경로
directory_path = './data/raw'

# 디렉토리에서 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
selected_file = st.selectbox("Choose a CSV file", csv_files)

# 선택한 CSV 파일 경로
csv_file_path = os.path.join(directory_path, selected_file)

# KLUE BERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

try:
    # 첫 번째 행을 헤더로 인식
    df = pd.read_csv(csv_file_path, header=0)

    # 랜덤 데이터 미리보기
    st.subheader("Random Data Preview")

    # 라벨과 샘플 수 입력받기
    label_input = st.text_input("Enter a label (optional) 빈칸 시 랜덤")
    sample_size_input = st.number_input("Enter the number of samples", min_value=1, max_value=len(df), value=5, step=1)

    # 샘플 표시 버튼 추가
    if st.button("Show Samples"):
        # 샘플 수 설정: 빈칸이면 기본값 5 적용
        sample_size = int(sample_size_input) if sample_size_input else 5

        # 라벨 입력 여부에 따른 데이터 필터링
        if label_input:
            # 특정 라벨 필터링
            label_data = df[df["target"] == int(label_input)]
            if not label_data.empty:
                st.write(label_data.sample(min(len(label_data), sample_size)))
            else:
                st.write(f"No data found with label {label_input}. Showing random samples instead.")
                st.write(df.sample(sample_size))
        else:
            # 라벨이 빈칸인 경우 전체 데이터에서 랜덤 샘플
            st.write(df.sample(sample_size))

    # 라벨 분포 시각화
    st.subheader("Label Distribution")
    label_counts = df["target"].value_counts()
    st.bar_chart(label_counts)

    # 텍스트 길이 분포
    st.subheader("Text Length Distribution")
    df['text_length'] = df['text'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(df['text_length'], bins=30, kde=True, ax=ax)
    ax.set_title("Text Length Distribution")
    st.pyplot(fig)

    # 결측값 확인
    st.subheader("Missing Values Check")
    st.write(df.isnull().sum())

    # 중복 데이터 확인
    st.subheader("Duplicate Data Check")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # 텍스트 길이와 라벨 간 상관 관계 분석
    st.subheader("Text Length by Label")
    fig, ax = plt.subplots()
    sns.boxplot(x="target", y="text_length", data=df, ax=ax)
    ax.set_title("Text Length by Label")
    st.pyplot(fig)

    # 토큰 수 분포 분석
    df['token_count'] = df['text'].apply(lambda x: len(tokenizer(x)["input_ids"]))
    st.subheader("Token Count Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['token_count'], bins=30, kde=True, ax=ax)
    ax.set_title("Token Count Distribution")
    st.pyplot(fig)

    # 입력한 ID의 뒷부분 숫자와 일치하는 데이터 검색
    st.subheader("Tokenization for Specific ID")
    input_number = st.text_input("Enter the last 5 digits of the ID (e.g., 00000):")
    
    if input_number:
        # ID 열에서 입력한 번호와 일치하는 데이터 필터링
        matching_data = df[df["ID"].str.endswith(input_number)]
        
        if not matching_data.empty:
            # 해당 데이터의 텍스트 가져오기
            selected_text = matching_data["text"].values[0]
            st.write(f"Text for ID ending in {input_number}:", selected_text)
            
            # 토크나이징 결과 확인
            tokenized_output = tokenizer(selected_text)
            st.write("Tokens:", tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"]))
        else:
            st.write("No data found with the specified ID ending.")
    
except FileNotFoundError:
    st.error(f"File not found at path: {csv_file_path}. Please check the path.")
