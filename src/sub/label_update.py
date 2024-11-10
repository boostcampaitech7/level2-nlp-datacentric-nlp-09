import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# CSV 파일 읽기
df = pd.read_csv('data/preprocessed/train_normal.csv')  # 'input_file.csv'는 실제 파일 이름으로 변경

df = df.head(3)

# 주제 레이블
topic_labels = ["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"]

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# 텍스트를 BERT 임베딩으로 변환하는 함수
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰의 출력값을 사용
    return outputs.last_hidden_state[:, 0, :].numpy()

# 주제 레이블을 BERT 임베딩으로 변환
topic_vectors = [embed_text(topic) for topic in topic_labels]
topic_vectors = np.vstack(topic_vectors)  # 2D 배열로 변환

# 각 문장에 대해 주제와의 유사도 계산
def find_closest_topic(text):
    # 문장을 임베딩
    text_vector = embed_text(text)
    # 주제 레이블과의 코사인 유사도 계산
    similarities = cosine_similarity(text_vector, topic_vectors)
    # 가장 유사한 주제의 인덱스 반환
    return similarities.argmax()

# target 컬럼 추가
df['target'] = df['text'].apply(find_closest_topic)

# 결과를 새로운 CSV 파일로 저장
df.to_csv('data/preprocessed/train_label_update.csv', index=False) 
