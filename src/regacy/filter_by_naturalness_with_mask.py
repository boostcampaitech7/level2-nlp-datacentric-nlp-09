# # src/preprocessing/preprocessing.py
# import pandas as pd
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# # 모델 및 토크나이저 초기화
# model_name = "klue/bert-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)
# model.eval()

# def calculate_naturalness(sentence):
#     """각 단어를 마스킹하여 자연스러움 점수를 계산하는 함수"""
#     input_ids = tokenizer.encode(sentence, return_tensors="pt")
#     naturalness_score = 0.0
#     num_tokens = len(input_ids[0])

#     # 각 단어를 마스킹하고 자연스러움 점수를 계산
#     for i in range(1, num_tokens - 1):  # [CLS], [SEP] 제외
#         masked_input_ids = input_ids.clone()
#         masked_input_ids[0, i] = tokenizer.mask_token_id  # 토큰 마스킹

#         with torch.no_grad():
#             outputs = model(masked_input_ids)
#             logits = outputs.logits

#         # 마스킹된 위치의 실제 토큰의 확률
#         true_token_id = input_ids[0, i]
#         token_prob = torch.softmax(logits[0, i], dim=-1)[true_token_id].item()
#         naturalness_score += torch.log(torch.tensor(token_prob))

#     # 평균 로그 확률 계산
#     return (naturalness_score / num_tokens).item()

# def add_naturalness_scores(df, text_column="text"):
#     """데이터프레임에 자연스러움 점수를 추가"""
#     df['naturalness_score'] = df[text_column].apply(calculate_naturalness)
#     return df

# def save_with_naturalness_score(df, output_dir, filename):
#     """자연스러움 점수를 추가한 데이터를 저장"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     save_path = os.path.join(output_dir, f"with_naturalness_score_{filename}")
#     df.to_csv(save_path, index=False)
#     return save_path

# # 예시 실행
# if __name__ == "__main__":
#     # 파일 로드
#     input_path = "./data/preprocessed/processed_train_v1.csv"  # 예시 파일 경로
#     output_dir = "./data/preprocessed"
#     df = pd.read_csv(input_path)
    
#     # 자연스러움 점수 추가
#     df_with_scores = add_naturalness_scores(df)
    
#     # 파일 저장
#     save_with_naturalness_score(df_with_scores, output_dir, "processed_train_v2.csv")
#     print(f"Processed file saved to {output_dir}")

import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from multiprocessing import Pool, cpu_count

# 모델 및 토크나이저 초기화
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextDataset(Dataset):
    """텍스트 데이터셋 클래스"""
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def calculate_naturalness(sentence):
    """각 단어를 마스킹하여 자연스러움 점수를 계산하는 함수"""
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    naturalness_score = 0.0
    num_tokens = len(input_ids[0])

    for i in range(1, num_tokens - 1):  # [CLS], [SEP] 제외
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input_ids)
            logits = outputs.logits

        true_token_id = input_ids[0, i]
        token_prob = torch.softmax(logits[0, i], dim=-1)[true_token_id].item()
        naturalness_score += torch.log(torch.tensor(token_prob)).item()

    return naturalness_score / num_tokens

def process_batch(sentences):
    """배치 내 각 문장에 대해 자연스러움 점수를 계산"""
    return [calculate_naturalness(sentence) for sentence in sentences]

def add_naturalness_scores(df, text_column="text", batch_size=8):
    """DataLoader를 활용하여 배치 단위로 자연스러움 점수를 계산하여 데이터프레임에 추가"""
    texts = df[text_column].tolist()
    dataset = TextDataset(texts)  # 텍스트 데이터셋 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    naturalness_scores = []
    for batch in dataloader:
        scores = process_batch(batch)  # 배치 처리 함수 사용
        naturalness_scores.extend(scores)

    df['naturalness_score'] = naturalness_scores
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
    input_path = "./data/raw/train.csv"
    output_dir = "./data/preprocessed"
    df = pd.read_csv(input_path)
    
    # 자연스러움 점수 추가
    df_with_scores = add_naturalness_scores(df)
    
    # 파일 저장
    save_with_naturalness_score(df_with_scores, output_dir, "train_nat_bert.csv")
    print(f"Processed file saved to {output_dir}")

