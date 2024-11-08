from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_dir = "kfkas/t5-large-korean-P2G"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

text = "생r인증②_D안*a제bK 유@되면 대J _가끝"
inputs = tokenizer.encode(text, return_tensors="pt")
output = model.generate(inputs)

# Use decode() instead of batch()
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)
