import pandas as pd
import numpy as np
import os
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import re





BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data/preprocessed")
data1 = pd.read_csv(os.path.join(DATA_DIR, 'label_fix.csv'))


model_name = "skt/kogpt2-base-v2"
generator = pipeline("text-generation", model=model_name, device=0)
llm = HuggingFacePipeline(pipeline=generator)
prompt_template = PromptTemplate(
    input_variables=["keywords"],
    #template="{keywords}(으)로 스포츠 뉴스제목을 생성."
    template="{keywords}"
)
chain = LLMChain(llm=llm, prompt=prompt_template)

def contains_special_chars(text):
    return bool(re.search(r"[^a-zA-Z0-9\s.%가-힣\u4E00-\u9FFF~]", text))

for idx, row in data1.iterrows():
    if contains_special_chars(row["text"]):  # 특수문자 확인
        keywords = row["nouns"]
        headline = chain.run(keywords=keywords)
        # 뉴스 제목 생성
        headlin = headline.split(".")
        if len(headlin) > 1:
            headline = headlin[1]
        else:
            headline = headlin[0]       
        data1.at[idx, "text"] = headline  # "text" 컬럼 값 업데이트
        
# '\n' 문자 제거
data1['text'] = data1['text'].str.replace('\n', '', regex=False)
data1['text'] = data1['text'].str.replace('"', '', regex=False).str.replace("'", '', regex=False)

data1.to_csv("data/preprocessed/text_fix_LLM.csv", index=False, encoding='utf-8-sig')