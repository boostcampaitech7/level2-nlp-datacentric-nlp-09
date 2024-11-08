from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()

INPUT_DIR = 'data/preprocessed_v2/Lnoise_Augmentation_4981.csv'
OUTPUT_DIR = 'data/preprocessed_v2/Lnoise_Augmentation_4981_aya-Lrecovered.csv'
model_id = "CohereForAI/aya-expanse-32b"
device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_topic(model, tokenizer, prompt, text):
    input_ids = tokenizer.encode(prompt + text, return_tensors="pt").to(device)
    gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=15, 
    do_sample=True,
    temperature=0.3,  # 다양성을 높이기 위해 1.0보다 큰 값
    # top_p=0.9,  # nucleus sampling
    # top_k=50,  # 상위 50개 단어 중 샘플링
    # no_repeat_ngram_size=2,
    # num_return_sequences=10,  # 한 번에 n개의 다른 응답 생성
    )
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    gen_text = gen_text.split(text)[1]
    return gen_text


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


prompt = """ 다음은 뉴스 제목들입니다. 각 제목이 어떤 Topic에 속하는지를 분석하여 예측해 주세요. Topic은 예를 들어 '정치', '스포츠' 등과 같이 일반적인 카테고리를 포함합니다. 각 제목을 보고 모델이 가장 적합한 Topic을 제시하세요.

제목:

제목 1: 남북 관계 개선, 국제 사회 긍정적 평가 이어져
제목 2: 메시·호날두 UEFA 올해의 팀에 선정…EPL 선수 제로

출력 형식:

Topic 1: 정치
Topic 2: 스포츠

위와 같은 형식으로 각 제목에 맞는 Topic을 예측해 주세요.
제목 : """

df = pd.read_csv(INPUT_DIR)
df['topic'] = df['text'].progress_apply(lambda x: predict_topic(model, tokenizer, prompt, x))
df.to_csv(OUTPUT_DIR, index=False)