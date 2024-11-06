import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def sample_text(df, num_samples=30):
    df_sampled = df.sample(num_samples, random_state=456)
    text_list = df_sampled['text'].tolist()
    return text_list

def sample_text_template():
    return """
    한국 경제, 3분기 성장률 0.3%로 둔화
    대선 후보들, 경제 정책 발표로 표심 공략
    부산, 폭우로 인한 피해 복구 작업 본격 시작
    기후 변화 대응을 위한 국제 회의 개최
    한국 연구팀, 인공지능 기반 신약 개발 성공
    국내 IT 기업, 메타버스 플랫폼 출시 예정
    2024 서울국제영화제, 개막작 공개
    한국 대표팀, 아시안게임에서 금메달 획득
    전국 대학, AI 관련 전공 신설 이어져
    유명 배우, 새로운 드라마 출연 소식 전해
    전문가들, 건강한 식습관의 중요성 강조
    학교 폭력 예방을 위한 정부의 새로운 정책 발표
    부동산 시장, 가격 하락세 지속
    여당, 신규 정책으로 국민 지지율 상승
    플라스틱 사용 줄이기 위한 캠페인 전개
    AI 기술, 제조업 혁신의 핵심으로 부상
    우주 탐사, 새로운 행성 발견 보고
    한국 전통 음악, 세계적으로 주목받다
    프로야구, 포스트시즌 진출팀 확정
    소비자 물가 상승률, 10년 만에 최고치 기록
    야당, 정부의 부동산 정책 비판
    아동학대 예방을 위한 법 개정안 통과
    미세먼지 저감을 위한 새로운 대책 발표
    로봇 기술, 의료 분야에 적용 확대
    유전자 편집 기술, 농업 혁신에 기여
    한국 문화재, 유네스코 세계유산으로 등재
    올림픽, 선수촌 시설 점검 완료
    온라인 학습, 새로운 교육 트렌드로 자리잡아
    가수, 새 앨범 발매 예고하며 팬들 기대
    정신 건강 관리, 사회적 관심 증가
    """



# Function to create the prompt with few-shot examples
def create_prompt(text_list):
    prompt = (
        "너는 숙련된 뉴스 작성자로서, 뉴스 도메인에 대한 지식이 있어야 한다.\n"
        "너는 뉴스 도메인에 대한 지식을 바탕으로 뉴스 제목을 생성할 수 있다.\n"
        "뉴스 제목을 생성해라.\n예시: "
    )
    prompt += text_list + "\n답변: "
    
    return prompt


# 토크나이저와 모델 로드
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df_GTandLnoise = pd.read_csv('data/preprocessed/asterisk2GTandLnoise.csv')

# List to store generated texts
generated_texts = []

# Create the prompt
prompt = create_prompt(sample_text_template())

# Tokenize the prompt and move inputs to the correct device
inputs = tokenizer(prompt, return_tensors='pt').to(device)

# Generate the output using the model
outputs = model.generate(
    **inputs,
    max_new_tokens=50,          # Adjust based on expected output length
    num_beams=5,
    # temperature=0.2,
    # top_p=0.85,
    do_sample=False,  # 샘플링 활성화
    no_repeat_ngram_size=2,
)

# Decode the generated tokens
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the generated text after '답변:'
# generated_output = output_text.split('답변:')[-1].strip()
# generated_output = generated_output.split('\n')[0].strip()  
# Append to the list
# generated_texts.append(generated_output)


# # Add the generated texts to the DataFrame
# make a new empty DataFrame
# df = df_GTandLnoise.head(1).copy()

# df['ID'] = 'Augmented'
# df['text'] = generated_texts
# df['target'] = -1

print(prompt)

print(output_text)

# for i in range(df.shape[0]):
#     print(f"{df_test.at[i, 'ID']}\n{df_test.at[i, 'text']}\n{df_test.at[i, 'text_gen']}\n")
    
# # Save the updated DataFrame to a new CSV file
# df.to_csv('data/preprocessed/asterisk2recover_Tnoise.csv', index=False)
# df_test.to_csv('data/preprocessed/asterisk2recover_Tnoise_test.csv', index=False)