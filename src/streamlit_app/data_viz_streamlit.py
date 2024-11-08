import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import os
import re

# CSV 파일이 있는 디렉토리 경로
directory_path = './data/preprocessed'

# 디렉토리에서 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
selected_file = st.selectbox("Choose a CSV file", csv_files)

# 선택한 CSV 파일 경로
csv_file_path = os.path.join(directory_path, selected_file)

# KLUE BERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

percentiles = [0.1, 0.2, 0.8, 0.9]

try:
    # 첫 번째 행을 헤더로 인식
    df = pd.read_csv(csv_file_path, header=0)

    # 특수문자 입력 필드 추가
    st.subheader("Filter by Special Character")
    special_char_input = st.text_input("Enter special characters to filter (e.g., @#$)")

    # ASCII 비율 라벨과 샘플 수 입력받기
    label_input = st.text_input("Enter a minimum ASCII ratio (optional)")
    sample_size_input = st.number_input("Enter the number of samples", min_value=1, max_value=len(df), value=5, step=1)

    # 샘플 표시 버튼 추가
    if st.button("Show Samples"):
        # 샘플 수 설정: 빈칸이면 기본값 5 적용
        sample_size = int(sample_size_input) if sample_size_input else 5
        
        # 특수문자가 입력된 경우 해당 문자가 포함된 행만 필터링
        if special_char_input:
            # 각 특수문자를 OR 조건으로 연결하여 필터링
            special_chars = f"[{re.escape(special_char_input)}]"
            filtered_df = df[df['text'].str.contains(special_chars)]
        else:
            filtered_df = df  # 특수문자가 입력되지 않으면 전체 데이터 사용
        
        # ASCII 비율 기준 필터링
        if label_input:
            filtered_df = filtered_df[filtered_df["ascii_ratio"] >= float(label_input)]
        
        # 필터링된 데이터에서 샘플링
        if not filtered_df.empty:
            st.write(filtered_df.sample(min(len(filtered_df), sample_size)))
        else:
            st.write("No data found with the specified filters.")
        
        # 필터링된 데이터의 총 개수 출력
        st.write(f"Number of rows within selected range: {len(filtered_df)}")

    
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
    
    # 특수문자 비율에 대한 분포도 시각화
    st.subheader("Distribution of ASCII Character Ratio (33-126)")
    fig, ax = plt.subplots()
    sns.histplot(df['ascii_ratio'], bins=30, kde=True, ax=ax)
    ax.set_title("ASCII Character Ratio Distribution (33-126)")
    ax.set_xlabel("ASCII Character Ratio")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 텍스트 길이와 라벨 간 상관 관계 분석
    st.subheader("Text Length by Label")
    fig, ax = plt.subplots()
    sns.boxplot(x="target", y="text_length", data=df, ax=ax)
    ax.set_title("Text Length by Label")
    st.pyplot(fig)
    df_description = df['text_length'].describe(percentiles=percentiles)
    st.write("Text Length Statistics:")
    st.write(df_description)

    # 토큰 수 분포 분석
    df['token_count'] = df['text'].apply(lambda x: len(tokenizer(x)["input_ids"]))
    st.subheader("Token Count Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['token_count'], bins=30, kde=True, ax=ax)
    ax.set_title("Token Count Distribution")
    st.pyplot(fig)
    df_description = df['token_count'].describe(percentiles=percentiles)
    st.write("Token Count Statistics:")
    st.write(df_description)

    # 텍스트 또는 토큰 길이 필터링 옵션 추가
    st.subheader("Filter by Text or Token Length")
    filter_option = st.radio("Choose length type to filter by:", ('Text Length', 'Token Length'))

    if filter_option == 'Text Length':
        min_length, max_length = st.slider("Select text length range:", 
                                           int(df['text_length'].min()), 
                                           int(df['text_length'].max()), 
                                           (int(df['text_length'].quantile(0.1)), int(df['text_length'].quantile(0.9))))
        filtered_df = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
    else:
        min_length, max_length = st.slider("Select token length range:", 
                                           int(df['token_count'].min()), 
                                           int(df['token_count'].max()), 
                                           (int(df['token_count'].quantile(0.1)), int(df['token_count'].quantile(0.9))))
        filtered_df = df[(df['token_count'] >= min_length) & (df['token_count'] <= max_length)]

    # 필터링된 데이터 출력
    st.write(f"Number of rows within selected range: {len(filtered_df)}")
    st.write(filtered_df.head())

    # 입력한 ID의 뒷부분 숫자와 일치하는 데이터 검색
    st.subheader("Tokenization for Specific ID")
    input_number = st.text_input("Enter the last digits of the ID (e.g., 00000):")
    
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
