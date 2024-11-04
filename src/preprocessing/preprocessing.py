# preprocessing.py
import pandas as pd
from transformers import AutoTokenizer

# KLUE BERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 텍스트 및 토큰 길이 기준
TEXT_LENGTH_THRESHOLD = 16
TOKEN_LENGTH_THRESHOLD = 9

def load_data(file_path):
    """CSV 파일을 로드하는 함수"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """텍스트 및 토큰 길이 기준으로 데이터 필터링하는 함수"""
    # 텍스트 길이 및 토큰 길이 계산
    df['text_length'] = df['text'].apply(len)
    df['token_count'] = df['text'].apply(lambda x: len(tokenizer(x)["input_ids"]))
    
    # 기준에 따라 필터링된 데이터 반환
    filtered_df = df[(df['text_length'] > TEXT_LENGTH_THRESHOLD) & (df['token_count'] > TOKEN_LENGTH_THRESHOLD)]
    return filtered_df

def get_thresholds():
    """텍스트와 토큰 길이 기준 반환 함수"""
    return TEXT_LENGTH_THRESHOLD, TOKEN_LENGTH_THRESHOLD
