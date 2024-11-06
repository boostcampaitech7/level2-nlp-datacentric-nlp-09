import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np



DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BASE_DIR = os.getcwd()
TEST_DIR = os.path.join(BASE_DIR, 'data/preprocessed/mask2goldentruth.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/preprocessed/GT_label_restored.csv') # 해당 파일 이름으로 output 폴더 생성

# model_name = 'Doowon96/bert-base-finetuned-ynat' # ynat 불가능
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

dataset_test = pd.read_csv(TEST_DIR)

model.eval()
preds = []

for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
    inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
        preds.extend(pred)
        
dataset_test['target'] = preds
dataset_test.to_csv(OUTPUT_DIR, index=False) # processed된 파일 이름 폴더에 결과 저장