# src/preprocessing/preprocessing.py
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# KLUE BERT 토크나이저와 언어 모델 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")

# 텍스트 및 토큰 길이 기준
TEXT_LENGTH_THRESHOLD = 16
TOKEN_LENGTH_THRESHOLD = 9
NATURALNESS_THRESHOLD = -5.0  # 자연스러움 점수 임계값

def load_data(file_path):
    """CSV 파일을 로드하는 함수"""
    return pd.read_csv(file_path)

def calculate_naturalness(text):
    """문장의 자연스러움 점수를 계산하는 함수"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    token_probs = torch.softmax(logits, dim=-1)
    token_ids = inputs["input_ids"].squeeze()
    naturalness_score = 0.0

    for i, token_id in enumerate(token_ids):
        token_prob = token_probs[0, i, token_id]
        naturalness_score += torch.log(token_prob).item()

    return naturalness_score / len(token_ids)

def add_naturalness_score(df):
    """자연스러움 점수를 데이터프레임에 추가하는 함수"""
    # 자연스러움 점수 계산
    df['naturalness_score'] = df['text'].apply(calculate_naturalness)
    return df

def save_with_naturalness_score(df, output_dir, filename):
    """자연스러움 점수를 추가한 데이터를 지정된 디렉토리에 저장하는 함수"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"with_naturalness_score_{filename}")
    df.to_csv(save_path, index=False)
    return save_path

def get_thresholds():
    """텍스트와 토큰 길이, 자연스러움 점수 기준 반환 함수"""
    return {
        "text_length": TEXT_LENGTH_THRESHOLD,
        "token_length": TOKEN_LENGTH_THRESHOLD,
        "naturalness_score": NATURALNESS_THRESHOLD
    }
