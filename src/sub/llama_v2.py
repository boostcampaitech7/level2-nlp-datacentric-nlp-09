import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Llama 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")

def extract_keywords_and_add(original_text):
    # 프롬프트를 업데이트하여 원본 텍스트에서 중요한 키워드를 추출하도록 설정
    input_text = f"다음 문장에서 중요한 키워드만 추출해보세요:\n\n{original_text}\n\n추출된 키워드:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # 모델을 통해 키워드 추출
    outputs = model.generate(inputs, max_new_tokens=20, num_return_sequences=1)
    keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # '추출된 키워드:' 이후의 키워드만 반환하고, 원본 문장에 추가
    extracted_keywords = keywords.split("추출된 키워드:")[-1].strip()
    return f"{original_text} {extracted_keywords}"

# CSV 파일 경로
file_path = 'data/raw/train.csv'

# CSV 파일 읽기
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for i, row in enumerate(reader):
        # 상단 3개 항목만 읽기
        if i >= 3:
            break
        # 필요한 컬럼만 저장
        data.append({
            'id': row.get('ID'),
            'text': row.get('text'),
            'target': row.get('target')
        })

# 새 답변을 추가할 데이터 저장소
new_answers = []

# 각 항목에 대해 키워드를 추출하여 원본 텍스트에 추가
for item in data:
    enhanced_text = extract_keywords_and_add(item['text'])
    new_answers.append({
        'ID': item['id'],
        'text': enhanced_text,
        'target': item['target']
    })

# 결과를 CSV 파일로 저장
with open('data/preprocessed/train_llama_v2.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'text', 'target'])
    writer.writeheader()

    for row in new_answers:
        writer.writerow(row)
