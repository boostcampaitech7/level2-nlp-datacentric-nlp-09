import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import torch
import random
import numpy as np

# Seed 고정
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 원하는 시드 값 설정

# 데이터 로드
train_df = pd.read_csv("data/preprocessed/train_noise_v3.csv")
test_df = pd.read_csv("data/augmented/train_llm_v2.csv")

# 데이터셋 준비 (train-test split)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(), train_df['target'].tolist(), test_size=0.1, random_state=42
)

# Tokenizer와 데이터셋 생성
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)

train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels})
val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'], 'labels': val_labels})
test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']})

# 모델 초기화
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=len(set(train_labels)))

# 데이터 collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 메트릭 정의
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# 학습 설정
training_args = TrainingArguments(
    output_dir='data/preprocessed',
    evaluation_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=6,
    weight_decay=0.01,
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 테스트 데이터셋 예측
predictions = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1)

# 예측 결과 저장
test_df['target'] = pred_labels.numpy()
test_df.to_csv("data/augmented/train_llm_v3.csv", index=False)
