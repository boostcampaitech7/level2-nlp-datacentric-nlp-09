# src/preprocessing/preprocessing.py
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 모델 및 토크나이저 초기화
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

def calculate_naturalness(sentence):
    """각 단어를 마스킹하여 자연스러움 점수를 계산하는 함수"""
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    naturalness_score = 0.0
    num_tokens = len(input_ids[0])

    # 각 단어를 마스킹하고 자연스러움 점수를 계산
    for i in range(1, num_tokens - 1):  # [CLS], [SEP] 제외
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id  # 토큰 마스킹

        with torch.no_grad():
            outputs = model(masked_input_ids)
            logits = outputs.logits

        # 마스킹된 위치의 실제 토큰의 확률
        true_token_id = input_ids[0, i]
        token_prob = torch.softmax(logits[0, i], dim=-1)[true_token_id].item()
        naturalness_score += torch.log(torch.tensor(token_prob))

    # 평균 로그 확률 계산
    return (naturalness_score / num_tokens).item()

def add_naturalness_scores(df, text_column="text"):
    """데이터프레임에 자연스러움 점수를 추가"""
    df['naturalness_score'] = df[text_column].apply(calculate_naturalness)
    return df

def save_with_naturalness_score(df, output_dir, filename):
    """자연스러움 점수를 추가한 데이터를 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"with_naturalness_score_{filename}")
    df.to_csv(save_path, index=False)
    return save_path

# 예시 실행
if __name__ == "__main__":
    # 파일 로드
    input_path = "./data/raw/train.csv"  # 예시 파일 경로
    output_dir = "./data/preprocessed"
    df = pd.read_csv(input_path)
    
    # 자연스러움 점수 추가
    df_with_scores = add_naturalness_scores(df)
    
    # 파일 저장
    save_with_naturalness_score(df_with_scores, output_dir, "preprocessed_train.csv")
    print(f"Processed file saved to {output_dir}")
