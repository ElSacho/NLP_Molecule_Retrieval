from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
# from Model02 import Model
# from ModelOneHead import Model
from ModelOneHeadNLP import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from tqdm import tqdm
import csv

from lraps import calculate_val_lraps

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

# model_name = str(input("Please write the model name : "))
# model_name = 'distilbert-base-uncased'
# model_name = 'microsoft/MiniLM-L12-H384-uncased'
# model_name = 'microsoft/MiniLM-L6-H384-uncased'
model_name = 'allenai/scibert_scivocab_uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)

gt = np.load("data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 30
batch_size = 12
learning_rate = 3e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=600, graph_hidden_channels=300) # nout = bert model hidden dim
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1_000_000
best_lraps = 0

# Initialize the CSV file
with open('validation_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Validation Loss', 'LRAPS'])

# checkpoint = torch.load('model_checkpoint.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
torch.cuda.empty_cache() # test to liberate memory space
for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids') # This is likely done to prevent the input_ids from being processed in the subsequent graph operations, as they might be handled separately.
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
        torch.cuda.empty_cache() # test to liberate memory space
    model.eval()
    val_loss = 0        
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
        torch.cuda.empty_cache() # test to liberate memory space
    lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
    torch.cuda.empty_cache() # test to liberate memory space
    best_validation_loss = min(best_validation_loss, val_loss)
    best_lraps = max(best_lraps, lraps)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
    with open('validation_metrics.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, val_loss, lraps])
    if best_lraps==lraps:
        print('lraps loss improoved saving checkpoint...')
        save_path = os.path.join('./models', 'model_attention'+str(i)+'.pt')
        # model.to('cpu')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, 'model_checkpoint.pt')
        print('checkpoint saved to: {}'.format(save_path))
        # model.to(device)
        torch.cuda.empty_cache() # test to liberate memory space
    torch.cuda.empty_cache() # test to liberate memory space


print('loading best model...')
# checkpoint = torch.load(save_path)
checkpoint = torch.load('model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)
print("finished")