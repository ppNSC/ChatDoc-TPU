from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os

# list of sentences
sentences = ['如何更换花呗绑定银行卡', '算能2023年第四季度销售会议通报', '两化企业管理方法TKP', '比特家校招文化分享会通报']

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('../bce-embedding-base_v1')
model = AutoModel.from_pretrained('../bce-embedding-base_v1')

device = 'cpu'  # if no GPU, set "cpu"
model.to(device)

# get inputs
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()

if input_ids.shape[1] > 512:
    input_ids = input_ids[:, :512]
    attention_mask = attention_mask[:, :512]
elif input_ids.shape[1] < 512:
    input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=0)
    attention_mask = np.pad(attention_mask,
                            ((0, 0), (0, 512 - attention_mask.shape[1])),
                            mode='constant', constant_values=0)
input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)

# Compute token embeddings
with torch.no_grad():
    model_output = model(input_ids, attention_mask)

embeddings = model_output.last_hidden_state[:, 0]  # cls pooler
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
folder = './onnx'
if not os.path.exists(folder):
    os.makedirs(folder)

torch.onnx.export(model, (input_ids,attention_mask), "./onnx/bce-embedding-base_v1.onnx", input_names=['input_ids', 'attention_mask'],dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}})

print('export onnx done!')