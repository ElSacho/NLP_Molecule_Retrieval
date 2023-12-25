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

from torchvision import datasets, transforms

import json

from utils import calculate_val_lraps

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def train(list_config_path):
    best_lraps = 0
    for config_path in list_config_path:
        best_lraps = train_conf(config_path, best_lraps)

def train_conf(config_path, best_lraps):
    
    with open(config_path, 'r') as file:
        parameters = json.load(file)
    # parameters = json.load(config_path)
    print("parameters opened")
    model_name = parameters['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='data/', gt=gt, split='train', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_epochs = parameters['nb_epochs']
    batch_size = parameters['batch_size']

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model_path = parameters['model_path']
    spec = importlib.util.spec_from_file_location("ModelModule", model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["ModelModule"] = model_module
    spec.loader.exec_module(model_module)

    # Supposons que la classe du modèle s'appelle 'Model'
    Model = getattr(model_module, 'Model')

    model = Model(parameters) # nout = bert model hidden dim
    model.to(device)
    
    print("model uploaded")

    weight_decay = parameters['weight_decay']
    learning_rate = parameters['learning_rate']

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999),
                                    weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if parameters['load_model_path'] != "None":
        try : 
            checkpoint = torch.load(parameters['load_model_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            raise Exception("Model path from configs.json not found")
    print('Start training')
    best_lraps = train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps)
    return best_lraps

def train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = 50):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
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
                                                                            time2 - time1, loss/count_iter))

            if count_iter > parameters['n_batches_before_break_in_epochs'] and parameters['n_batches_before_break_in_epochs'] != -1:
                break
            torch.cuda.empty_cache() # test to liberate memory space
        loss = 0
        writer.add_scalar('Loss/train', loss, epoch)
        
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
        print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        if best_lraps==lraps:
            print('lraps loss improoved saving checkpoint...')
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
        
        torch.cuda.empty_cache() # test to liberate memory space
    return best_lraps