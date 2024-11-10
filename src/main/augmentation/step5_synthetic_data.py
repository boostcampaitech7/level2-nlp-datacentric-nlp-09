import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import re

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Llama 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct").to(device)

def generate_news_domains():
    # 뉴스 도메인 리스트를 생성하는 프롬프트
    input_text = "뉴스 기사의 주요 도메인 9가지만 나열해 주세요."
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # 모델을 통해 뉴스 도메인 리스트 생성
    outputs = model.generate(inputs, max_new_tokens=50, num_return_sequences=1)
    domains_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)

    # 생성된 뉴스 도메인 리스트 반환
    return domains_text.strip()

def generate_headline_with_domain(domain):
    # 주제를 포함한 프롬프트 설정
    input_text = f"{domain} 관련 뉴스 헤드라인을 작성해주세요:\n\n헤드라인:"
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # 모델을 통해 헤드라인 생성
    outputs = model.generate(
        inputs, 
        max_new_tokens=50, 
        num_return_sequences=1, 
        do_sample=True, 
        temperature=0.9,  # 다양성 증가
        top_k=50,         # 샘플링 다양성 증가
        top_p=0.85        # 누적 확률로 다양성 조절
    )

    generated_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)

    # '헤드라인:' 이후의 실제 생성된 헤드라인만 반환
    split_text = generated_text.split("헤드라인:")
    
    # '헤드라인:' 뒤에 내용이 없으면 None 반환
    if len(split_text) < 2 or not split_text[-1].strip():
        return None

    # 생성된 헤드라인 반환 (개행 전까지만)
    generated_headline = split_text[-1].strip().splitlines()[0].replace('"', '').strip()
    return generated_headline

# CSV 파일 경로
file_path = 'data/augmented/train_llm_v1.csv'

# 뉴스 도메인 리스트 생성
news_domains = generate_news_domains()

# 리스트 추출을 위한 정규 표현식
domains = [domain for domain in re.findall(r'"(.*?)"', news_domains) if domain != "환경"]

# 추출된 도메인 리스트 출력
print("추출된 뉴스 도메인 리스트:", domains)

# 새로 생성한 주제별 헤드라인을 저장할 데이터 저장소
new_headlines = []
new_headlines = set()  # 중복 제거를 위해 set으로 변경

for domain in domains:
    count = 0
    while count < 1000:  # 각 주제마다 1000개의 헤드라인 생성
        headline = generate_headline_with_domain(domain)
        if headline and headline not in new_headlines:
            combined_text = f"{domain}: {headline}"
            new_headlines.add(combined_text)
            count += 1

# CSV 파일로 저장
with open(file_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['text'])
    writer.writeheader()
    
    # 저장할 때 딕셔너리 형태로 변환
    for headline_text in new_headlines:
        writer.writerow({'text': headline_text})

# 추출된 도메인 리스트 출력
print("csv 파일이 생성되었습니다.")
