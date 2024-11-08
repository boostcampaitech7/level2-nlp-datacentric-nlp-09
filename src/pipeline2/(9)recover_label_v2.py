import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

"""
이 코드는 (8)recover_label_v1.py에서 복구한 clean_data_v1.csv 전체 데이터셋을 사용하여 다시 학습하고, Label Noise를 복구합니다.
이후 Text와 Label이 모두 복구된 데이터셋을 최종적으로 저장합니다.
최종적으로 

코드 설명:
1. 모델을 불러오고, fine-tuning을 위한 설정을 정의합니다.
2. 학습용 데이터셋(Text 복구 데이터셋)과 테스트 데이터셋(Label 복구가 필요한 데이터셋)을 불러옵니다.
3. `BERTDataset` 클래스를 정의하여 데이터를 토큰화하고 PyTorch 데이터셋으로 변환합니다.
4. `Trainer` 클래스를 사용해 모델을 학습하며, F1-score를 평가 지표로 설정합니다.
5. 학습된 모델로 테스트 데이터셋에 대해 예측을 수행하고, 예측 결과를 저장합니다.
6. 두 개의 데이터셋을 병합하여 최종 CSV 파일(`clean_data_v2.csv`)로 저장합니다.

이 코드로 데이터셋의 학습 및 예측, 데이터 병합을 통해 최종 결과를 생성할 수 있습니다.
"""

# Set random seed
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Set device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data/preprocessed/clean_data_v1.csv') # Text noise recovered data
TEST_DIR = os.path.join(BASE_DIR, 'data/preprocessed/Lnoise_and_augmented.csv') # Label noise data
train_name = os.path.splitext(os.path.basename(DATA_DIR))[0]
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/preprocessed/Lnoise_and_augmented_recover_v2') # Output directory
CLEAN_DIR = 'data/preprocessed/clean_data_v2.csv'

# Load model and tokenizer
model_name = "classla/multilingual-IPTC-news-topic-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 7 classes
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=7,
    ignore_mismatched_sizes=True 
).to(DEVICE)

# Load data
data = pd.read_csv(DATA_DIR)
dataset_train, dataset_valid = train_test_split(data, test_size=0.2, random_state=SEED)

# Define BERTDataset class
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)

# Convert data to BERTDataset    
data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)

# Define data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define compute_metrics function
f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

## for wandb setting
# os.environ['WANDB_DISABLED'] = 'true'

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_strategy='steps',
    eval_strategy='steps',
    save_strategy='steps',
    logging_steps=10,
    eval_steps=10,
    save_steps=100,
    save_total_limit=2,
    learning_rate= 8e-06, # model card suggested
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon=1e-08,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    per_device_train_batch_size=32, # model card suggested
    per_device_eval_batch_size=32,
    num_train_epochs=5, # model card suggested
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    seed=SEED
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Load test data
dataset_test = pd.read_csv(TEST_DIR)

# Evaluate the model on the test data
model.eval()
preds = []
for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
    inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
        preds.extend(pred)

# Save the predictions        
dataset_test['target'] = preds
dataset_test.to_csv(OUTPUT_DIR+'.csv', index=False)

# Save the final result
final_Tnoise = pd.read_csv('../../data/preprocessed/Tnoise_comp_recover_1073.csv')
final_Lnoise = pd.read_csv(OUTPUT_DIR+'.csv')
clear = pd.concat([final_Tnoise, final_Lnoise], ignore_index=True)
clear.to_csv(CLEAN_DIR, index=False)