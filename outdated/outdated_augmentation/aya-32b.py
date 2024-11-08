from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import torch
from tqdm import tqdm

OUTPUT_DIR = 'data/augmented/aya-32b_v2.csv'
model_id = "CohereForAI/aya-expanse-32b"
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit 양자화 적용
    bnb_4bit_compute_dtype=torch.float16,  # 연산에 사용할 데이터 타입 설정
    bnb_4bit_use_double_quant=True,  # 이중 양자화 사용
    bnb_4bit_quant_type="nf4"  # 양자화 유형 설정 (nf4 또는 fp4)
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    # device_map="auto"
).to(device)

model.eval()
# device = "cuda" if torch.cuda.is_available() else "cpu"

prompt = """다음은 한국어 뉴스 제목을 작성하는 과제야. 너는 숙련된 한국 뉴스 기자처럼 생각하며 최신 뉴스 트렌드와 보도 방식을 반영해 짧고 간결한 제목을 특이하게 작성해야 해.

작성 지침:
1. 주제와 세부사항은 최대한 특이하게 만들어. 정치, 경제, 사회, 문화 등 다양한 분야를 다룰 수 있어.
2. 제목은 15자에서 30자 사이로 작성해.
3. 독자의 흥미를 끌 수 있도록 매력적이고 주목할 만한 내용을 반영해.
4. 중립적이고 사실 기반의 어조를 유지해.

예시:
답: 국내 IT기업, 혁신 기술 발표
답: 서울, 대기질 개선 프로젝트 착수
답: 여자농구 신한은행 팀 최소 득점 기록 34점 불명예
답: 메시·호날두 UEFA 올해의 팀에 선정…EPL 선수 제로
답: 여당, 신규 정책으로 국민 지지율 상승

이제 특이한 한국어 뉴스 제목을 한글로 작성해봐. 단답 형식으로 추가 설명없이 답만 바로 대답해.
답: 
"""

# Format the message with the chat template
# messages = [{"role": "user", "content": prompt}]
# input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

df = pd.DataFrame({
    'ID': ['start'],
    'text': ['start'],
    'target': ['start'],
})
df.to_csv(OUTPUT_DIR, index=False)

for i in tqdm(range(1000), desc="Generating"):
    original_df = pd.read_csv(OUTPUT_DIR)
    
    gen_texts = set()
    for i in range(10):
        gen_tokens = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            do_sample=True, 
            temperature=1.2,  # 다양성을 높이기 위해 1.0보다 큰 값
            top_p=0.9,  # nucleus sampling
            top_k=50,  # 상위 50개 단어 중 샘플링
            no_repeat_ngram_size=2,
            # num_return_sequences=10,  # 한 번에 n개의 다른 응답 생성
            )

        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        gen_text = gen_text.split(prompt)[1]
        gen_texts.add(gen_text)
        
    gen_texts = list(gen_texts)

    df = pd.DataFrame({
        'ID': ['augmentation'] * len(gen_texts),  # ID 열을 'augmentation'으로 채움
        'text': gen_texts,  # text 열에 리스트 l 삽입
        'target': [-1] * len(gen_texts)  # target 열을 -1로 채움
    })

    print(df)
    df = pd.concat([original_df, df])
    df.to_csv(OUTPUT_DIR, index=False)
