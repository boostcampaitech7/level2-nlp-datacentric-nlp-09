import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 모델 및 토크나이저 초기화
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_embedding(text):
    """문장 또는 토큰의 임베딩 벡터 계산 함수 (GPU 사용)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰의 임베딩 벡터를 사용
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def calculate_similarity_info(sentence):
    """
    문장과 각 토큰 간 유사도 정보 계산 함수
    - 유사도 평균과 각 토큰의 유사도 리스트 반환
    """
    # 문장 임베딩 계산
    sentence_embedding = calculate_embedding(sentence)
    
    # 토큰 단위로 분할하고 각 토큰의 임베딩 계산
    tokens = tokenizer.tokenize(sentence)
    token_embeddings = [calculate_embedding(token) for token in tokens]
    
    # 문장 임베딩과 각 토큰 임베딩 간의 유사도 계산
    similarities = [
        cosine_similarity([sentence_embedding], [token_embedding])[0][0]
        for token_embedding in token_embeddings
    ]
    
    # 유사도 평균과 유사도 리스트 반환
    return sum(similarities) / len(similarities), similarities

def add_similarity_info(df, text_column="text"):
    """데이터프레임에 유사도 평균과 토큰별 유사도 리스트를 추가하는 함수"""
    df['similarity_mean'], df['similarity_list'] = zip(*df[text_column].apply(calculate_similarity_info))
    return df

# 예시 실행
if __name__ == "__main__":
    # 파일 로드
    input_path = "./data/preprocessed/preprocessed_train.csv"  # 예시 파일 경로
    output_path = "./data/preprocessed/with_similarity_info_train_gpu.csv"
    df = pd.read_csv(input_path)
    
    # 유사도 평균 및 토큰별 유사도 리스트 추가
    df = add_similarity_info(df)
    
    # 파일 저장
    df.to_csv(output_path, index=False)
    print(f"Processed file with similarity information saved to {output_path}")
