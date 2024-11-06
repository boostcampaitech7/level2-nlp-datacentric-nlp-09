import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# Function to create the prompt with few-shot examples
def create_prompt(text, examples):
    prompt = (
        "다음은 '*'로 가려진 한국어 뉴스 기사 제목이야.\n"
        "주어진 뉴스 기사 제목 중 '*'로 가려져 있지 않는 부분을 활용하여 숙련된 한국 뉴스 기자가 쓸 법한 한국어 뉴스 기사 제목을 반말로 생성해.\n"
    )
    for i, ex in enumerate(examples):
        prompt += f"예시 {i+1}\n질문: {ex['input']}\n답변: {ex['output']}\n\n"
    prompt += f"질문: {text}\n답변:"
    return prompt


# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.5.3")
model = AutoModelForCausalLM.from_pretrained("jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.5.3")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv('data/preprocessed/noise2asterisk.csv')

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
    {
        'input': '홍* 시위 *여 여* 경찰**서 알몸 ** 강**해',
        'output': '홍콩 시위 참여 여성 경찰서에서 알몸 수색 강요당해'
    },
    {
        'input': '*북민 착한*사단 연천* 통일염* 벽화 *린다',
        'output': '탈북민 착한봉사단 연천서 통일염원 벽화 그린다'
    },
    {
        'input': '위안* 피해자 *림 *생님이 전하는 *야기',
        'output': '위안부 피해자 그림 선생님이 전하는 이야기'
    },
    {
        'input': '문 대*령 안*까움 금할 수 없어*사망* 최소화 만전 기*라',
        'output': '문 대통령 안타까움 금할 수 없어 사망자 최소화 만전 기하라'
    },
    {
        'input': '네이* 전*자료 검색 내* 안에 구* 따라** 것',
        'output': '네이버 전문자료 검색 내년 안에 구글 따라잡을 것'
    },
    {
        'input': '버** 튜브* 만든 설치*술',
        'output': '버려진 튜브로 만든 설치미술'
    },
    {
        'input': '과**회 대통령상* **과학고생 최경준*손승연',
        'output': '과학전람회 대통령상에 충북과학고생 최경준,손승연'
    },
    {
        'input': '베스트셀러*가벼운*에세이류*강세*계속',
        'output': '베스트셀러 가벼운 에세이류 강세 계속'
    },
    {
        'input': '스마트폰* **카드 대면 삑 새 ** 인증 나온*',
        'output': '스마트폰에 신용카드 대면 삑 새 본인 인증 나온다'
    },
    {
        'input': '이해찬 ** 취임 ** 난 앞에 놓고 인사*',
        'output': '이해찬 대표 취임 축하 난 앞에 놓고 인사말'
    },
    {
        'input': '** 시위에 원격** **폭탄 등장*경찰 테러**** 비슷',
        'output': '홍콩 시위에 원격조종 사제폭탄 등장,경찰 테러리스트와 비슷'
    },
    {
        'input': '특징주 수산아이앤티 코** 상장 *날 상한가**',
        'output': '특징주 수산아이앤티 코스닥 상장 첫날 상한가종합'
    },
    {
        'input': '이*** 이미 숨진 팔레***인 총격 **자 *까지 파괴',
        'output': '이스라엘 이미 숨진 팔레스타인인 총격 용의자 집까지 파괴'
    },
    {
        'input': '게시판 자랑*** 한양언론인상에 임**,이**',
        'output': '게시판 자랑스러운 한양언론인상에 임철순,이숙영'
    },
    {
        'input': '현대판 파라오 엘시시 *집* *지* 독**론도 겁*',
        'output': '현대판 파라오 엘시시 이집트 마지막 독립언론도 겁박'
    }
]


# List to store generated texts
generated_texts = []
original_outputs = []

# Process each row in the DataFrame
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating texts"):
    # Create the prompt
    prompt = create_prompt(row['text'], few_shot_examples)
    
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
    generated_output = output_text.split('답변:')[-1].strip()
    generated_output = generated_output.split('\n')[0].strip()
    
    # Append to the list
    generated_texts.append(generated_output)

# # Add the generated texts to the DataFrame
# df['text_gen'] = generated_texts
# df['text_gen_original'] = original_outputs

# save
df_test = df.copy()
df_test['text_gen'] = generated_texts
df['text'] = generated_texts

for i in range(df.shape[0]):
    print(f"{df_test.at[i, 'ID']}\n{df_test.at[i, 'text']}\n{df_test.at[i, 'text_gen']}\n")
    
# Save the updated DataFrame to a new CSV file
df.to_csv('data/preprocessed/asterisk2recover_Tnoise.csv', index=False)
df_test.to_csv('data/preprocessed/asterisk2recover_Tnoise_test.csv', index=False)