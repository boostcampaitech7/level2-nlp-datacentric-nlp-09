import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 분석할 CSV 파일이 있는 디렉토리 경로
directory_path = './data/preprocessed'

# 디렉토리에서 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
selected_file = st.selectbox("Choose a CSV file with naturalness scores", csv_files)

# 선택한 CSV 파일 경로
csv_file_path = os.path.join(directory_path, selected_file)

# 데이터 로드 및 자연스러움 점수 분포 분석
try:
    df = pd.read_csv(csv_file_path)

    if 'naturalness_score' in df.columns:
        st.subheader("Naturalness Score Analysis")

        # 자연스러움 점수 통계 분석
        st.write("Basic Statistics:")
        st.write(df['naturalness_score'].describe())

        # 자연스러움 점수 분포 히스토그램
        st.subheader("Naturalness Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['naturalness_score'], bins=30, kde=True, ax=ax)
        ax.set_title("Naturalness Score Distribution")
        ax.set_xlabel("Naturalness Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # 자연스러움 점수의 박스 플롯
        st.subheader("Naturalness Score Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['naturalness_score'], ax=ax)
        ax.set_title("Naturalness Score Boxplot")
        ax.set_xlabel("Naturalness Score")
        st.pyplot(fig)
        
        # 사용자 지정 범위에 따른 랜덤 샘플 확인
        st.subheader("Random Sample by Naturalness Score Range")
        
        # 자연스러움 점수 범위 슬라이더 추가
        min_score, max_score = df['naturalness_score'].min(), df['naturalness_score'].max()
        score_range = st.slider("Select Naturalness Score Range", min_value=float(min_score), max_value=float(max_score), value=(min_score, max_score))

        # 선택된 범위 내 데이터 필터링 및 갯수 표시
        filtered_df = df[(df['naturalness_score'] >= score_range[0]) & (df['naturalness_score'] <= score_range[1])]
        st.write(f"Number of rows within selected range: {len(filtered_df)}")

        # 랜덤 샘플 표시
        sample_size = st.number_input("Sample Size", min_value=1, max_value=len(filtered_df), value=5, step=1)
        st.write("Random Sample within Selected Range:")
        st.write(filtered_df.sample(n=sample_size))

    else:
        st.error("The selected file does not contain a 'naturalness_score' column.")

except FileNotFoundError:
    st.error(f"File not found at path: {csv_file_path}. Please check the path.")
except Exception as e:
    st.error(f"An error occurred: {e}")
