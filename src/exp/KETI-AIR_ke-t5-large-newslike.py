import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re

# Function to create the prompt with few-shot examples
def create_prompt(text, examples):
    prompt = (
        "다음은 '*'로 가려진 한국어 뉴스 기사 제목이야.\n"
        "주어진 뉴스 기사 제목 중 '*'로 가려져 있지 않는 부분을 활용하여 숙련된 한국 뉴스 기자가 쓸 법한 한국어 뉴스 기사 제목을 반말로 생성해.\n"
    )
    # prompt = (
    #     "다음은 '*'로 가려진 한국어 뉴스 기사 제목이야.\n"
    #     "주어진 뉴스 기사 제목 중 '*'로 가려져 있지 않는 부분을 의미적으로 최대한 복구하여 숙련된 한국 뉴스 기자가 쓸 법한 한국어 뉴스 기사 제목을 반말로 생성해.\n"
    # )
    for i, ex in enumerate(examples):
        prompt += f"예시 {i+1}\n질문: {ex['input']}\n답변: {ex['output']}\n\n"
    prompt += f"질문: {text}\n답변:"
    return prompt

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("t5-large")  # T5 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv('data/preprocessed_v2/Tnoise_comp_1073.csv').head(5)
sub = '*'
df['text'] = df['text'].apply(lambda x: re.sub(r'[^가-힣\s]', sub, x))

# Define few-shot examples
few_shot_examples = [
    {
        'input': '코스*지수 **00 돌파',
        'output': '코스피지수 3000 돌파'
    },
    {
        'input': '한**라운드 개막* 경기',
        'output': '한일라운드 개막전 경기'
    },
    {
        'input': '대**국 *선 후보 토*',
        'output': '대한민국 대선 후보 토론'
    },
    {
        'input': '여자농* 신한*행 팀 *소 득점 ** **점 불명*',
        'output': '여자농구 신한은행 팀 최소 득점 기록 34점 불명예'
    },
    {
        'input': '메*·호*두 U**A *해의 ** 선정…EP* 선수 제로',
        'output': '메시·호날두 UEFA 올해의 팀에 선정…EPL 선수 제로'
    },
]

# List to store generated texts
generated_texts = []

# Process each row in the DataFrame
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating texts"):
    # Create the prompt
    # prompt = create_prompt(row['text'], few_shot_examples)
    prompt = '뉴스 제목을 생성해봐:'
    
    # Tokenize the prompt and move inputs to the correct device
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Generate the output using the model
    outputs = model.generate(
        **inputs,
        max_length=1024,           # Adjust based on expected output length
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the generated tokens
    generated_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated text after '답변:'
    # generated_output = generated_output.split('답변:')[-1].strip()
    # generated_output = generated_output.split('\n')[0].strip()
    print(generated_output)
    
    # Append to the list
    generated_texts.append(generated_output)

# Add the generated texts to the DataFrame
df['text'] = generated_texts

# Save the updated DataFrame to a new CSV file
df.to_csv('data/preprocessed_v2/Tnoise_comp_ke_t5_1073.csv', index=False)
