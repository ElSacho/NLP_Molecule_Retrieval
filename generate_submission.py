import torch
import torch.nn.functional as F

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
# from ModelOneHeadNLP import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time
import os
import pandas as pd
from tqdm import tqdm
import csv

import importlib.util
import sys

# from models.disciminator import Discriminator
from losses import compute_triplet_loss, wgan_gp_loss, triplet_loss_sim
from torchvision import datasets, transforms
from sklearn.metrics.pairwise import cosine_similarity

import json

from utils import calculate_val_lraps, calculate_val_lraps_VQ

def make_csv_online(model, test_loader, test_text_loader, device, name=None):
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    
    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())
            
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)

    solution['ID'] = solution.index

    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]

    if name == None:
        solution.to_csv('submission.csv', index=False)
    else:
        solution.to_csv(f'{name}.csv', index=False)

def generate_csv(config_path):
    with open(config_path, 'r') as file:
        parameters = json.load(file)
    # parameters = json.load(config_path)
    print("parameters opened")
    model_name = parameters['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = parameters['batch_size']

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model_path = parameters['model_path']
    spec = importlib.util.spec_from_file_location("ModelModule", model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["ModelModule"] = model_module
    spec.loader.exec_module(model_module)

    # Supposons que la classe du modèle s'appelle 'Model'
    Model = getattr(model_module, 'Model')

    model = Model(parameters) # nout = bert model hidden dim
    model.to(device)
    
    checkpoint = torch.load("model_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
    print("The LRAPS on the val dataset is : ", lraps)
    
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    
    test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    for batch in tqdm(test_text_loader):
        for output in text_model(batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)

    solution['ID'] = solution.index

    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]

    solution.to_csv('submission.csv', index=False)
    print('finished !')
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <file_path>")
        sys.exit(1)

    # Le premier argument après le nom du script est le file_path
    file_path = sys.argv[1]

    generate_csv(file_path)