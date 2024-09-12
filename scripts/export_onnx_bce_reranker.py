import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os


# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('../bce-reranker-base_v1')
model = AutoModelForSequenceClassification.from_pretrained('../bce-reranker-base_v1')
sentence_pairs = [['apples', 'I like apples'],['apples', 'I like oranges'],['apples', 'Apples and oranges are fruits']]

device = 'cpu'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()
if input_ids.shape[1] > 512:
    input_ids = input_ids[:, :512]
    attention_mask = attention_mask[:, :512]
elif input_ids.shape[1] < 512:
    input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=1)
    attention_mask = np.pad(attention_mask,
                            ((0, 0), (0, 512 - attention_mask.shape[1])),
                            mode='constant', constant_values=0)
    
folder_name = "onnx"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)

model_predictions = model(input_ids, attention_mask, return_dict=True)
model_predictions = model_predictions.logits.view(-1,).float()

torch.onnx.export(model, (input_ids,attention_mask), "./onnx/bce-reranker-base_v1.onnx", input_names=['input_ids', 'attention_mask'], dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}})


# calculate scores
scores = torch.sigmoid(model_predictions)
print(scores)
print('export onnx done!')