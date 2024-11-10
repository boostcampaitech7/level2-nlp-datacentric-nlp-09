import pandas as pd
import numpy as np
import os
from konlpy.tag import Okt
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

'''


'''
def bring_noun_from_sentence(data) :
    okt =Okt()
    data['nouns'] = data['text'].apply(lambda x: okt.nouns(x))
    data['nouns'] = data['nouns'].apply(lambda x: ', '.join(x))
    return data
   
def get_word_embedding_for_MAX(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 마지막 히든 스테이트의 [CLS] 토큰 임베딩을 사용
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding
   


def get_embedding(word, tokenizer, model, device ):
    inputs = tokenizer(word, return_tensors="pt").to(device)  # 입력을 GPU로 이동
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 결과를 CPU로 이동   
 
 
# 유사도가 지정한 기준에 맞는 단어 쌍과 유사도 값을 중복 없이 리스트로 반환
def get_similar_pairs_with_new_word(similarity_matrix, words, new_word, new_word_similarities, threshold):
    similar_pairs = []  # 단어 쌍과 유사도 값을 담을 리스트
    for i, word in enumerate(words):
        if new_word_similarities[i] > threshold:
            #  해당 단어 간 유사도 값과 함께 단어 쌍 저장
            similar_pairs.append((new_word, word, new_word_similarities[i]))
    return similar_pairs
    
def fix_label(num, data) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # KLUE RoBERTa 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    model = AutoModel.from_pretrained("klue/roberta-base").to(device)  # 모델을 GPU로 이동
    #
    target_5 = data[data['target'] == num]
    target_5['text'] = target_5['text'].str.replace('…', ' ', regex=False)
    target_5['text'] = target_5['text'].str.replace('·', ' ', regex=False)
    target_5_filtered = target_5[target_5['text'].str.contains(r'[^a-zA-Z0-9\s.%가-힣\u4E00-\u9FFF]', regex=True)]
    
    ##
    noun_list = target_5_filtered['nouns'].unique().tolist()
    noun_list = [item.strip() for sublist in noun_list for item in sublist.split(',') if len(item.strip()) > 1]
    
    ###
    words = noun_list
      

    embeddings_for_MAX = [get_word_embedding_for_MAX(word, tokenizer, model) for word in words]
    cosine_similarities = cosine_similarity([embedding.numpy() for embedding in embeddings_for_MAX])
    similarity_sums = cosine_similarities.sum(axis=1)
    max_index = similarity_sums.argmax()
    max_word = words[max_index]
    
    ####

    words = noun_list
    new_word = max_word     
    embeddings = np.array([get_embedding(word) for word in words])
    new_word_embedding = get_embedding(new_word)     
    similarity_matrix = cosine_similarity(embeddings)
    new_word_similarities = cosine_similarity([new_word_embedding], embeddings).flatten()
    threshold = 0.96
    result_pairs = get_similar_pairs_with_new_word(similarity_matrix, words, new_word, new_word_similarities, threshold)
    result_words = set()
    for pair in result_pairs:
        result_words.add(pair[1]) 
    result_words = list(result_words) 
    
    
    #####
    target_5['target'] = target_5['text'].apply(lambda x: num if any(word in x for word in result_words) else 7)

    return target_5, result_words 




def assign_target(text, result):
    if any(word in text for word in result[0]):
        return 0
    elif any(word in text for word in result[1]):
        return 1
    elif any(word in text for word in result[2]):
        return 2
    elif any(word in text for word in result[3]):
        return 3
    elif any(word in text for word in result[4]):
        return 4
    elif any(word in text for word in result[5]):
        return 5
    elif any(word in text for word in result[6]):
        return 6
    else:
        return 7



def fix_extra_label(df_combined, result) :
    sub_7 = df_combined[df_combined['target'] == 7]
    df_combined = df_combined[df_combined['target'] != 7]
    df_combined = df_combined.reset_index(drop=True)
    sub_71 = sub_7[~sub_7['text'].str.contains(r"[^a-zA-Z0-9가-힣\s]", na=False)] 
    sub_71['target'] = sub_71['text'].apply(assign_target(result = result))
    sub_71 = sub_71[sub_71['target'] != 7]
    df_combined = pd.concat([df_combined, sub_71], ignore_index=True)
    
    return df_combined
    




BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

data = bring_noun_from_sentence(data)

result = []

df_combined = pd.DataFrame(columns=["ID",	"text",	"target"	,"nouns"])

for i in range(7) :
  target_df, result_words = fix_label(i, data)
  df_combined = pd.concat([df_combined, target_df], ignore_index=True)
  result.append(result_words)

fix_extra_label(df_combined, result)

df_combined.to_csv("data/preprocessed/label_fix.csv", index=False, encoding='utf-8-sig')
  