import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Llama 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")

def generate_clean_text(noisy_text):
    # 노이즈가 있는 텍스트를 정제하는 프롬프트 생성
    input_text = f"다음 문장에서 노이즈를 제거하고 정상 문장으로 출력하세요:\n\n{noisy_text}\n\n정제된 문장:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # 모델을 통해 정제된 텍스트 생성
    outputs = model.generate(inputs, max_new_tokens=50, num_return_sequences=1)
    cleaned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # '정제된 문장:' 이후의 실제 정제된 텍스트만 반환
    return cleaned_text.split("정제된 문장:")[-1].strip()

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

# 각 항목에 대해 정제된 텍스트 생성
for item in data:
    cleaned_text = generate_clean_text(item['text'])
    new_answers.append({
        'ID': item['id'],
        'text': cleaned_text,
        'target': item['target']
    })

# 결과를 CSV 파일로 저장
with open('data/preprocessed/train_llama_v1.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'text', 'target'])
    writer.writeheader()

    for row in new_answers:
        writer.writerow(row)
