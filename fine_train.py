import torch
import torch.nn.functional as F

from dataloader import GraphDataset, TextDataset
from dataloaderFt import GraphTextDatasetFineTune
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

from models.oneHead_LinearNLP import Model

from make_new_dataset import make_new_dataset

# from models.disciminator import Discriminator
from losses import wgan_gp_loss, triplet_loss_sim, contrastive_loss, cosine_similarity_loss
from torchvision import datasets, transforms

import json

from utils import calculate_val_lraps, calculate_val_lraps_VQ, print_parameters, calculate_val_lraps_text
from generate_submission import make_csv_online


def train(list_config_path):
    best_lraps = 0
    for config_path in list_config_path:
        make_new_dataset([config_path])
        best_lraps = train_conf(config_path, best_lraps)

def train_conf(config_path, best_lraps):
    
    with open(config_path, 'r') as file:
        parameters = json.load(file)
        
    parameters['batch_size'] = 3_000
    parameters['margin_delta'] = 0.8
    parameters['lambda_contra'] = 0.0
    parameters["print_every"] =  5
    parameters["expt_name"] += "f_t_new_dataset"
    parameters["log_dir"] += "f_t_new_dataset"
    parameters["load_model_path"] = parameters["save_path"]

    model_name = parameters['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDatasetFineTune(root='../data/', gt=gt, split='val',parameters= parameters, tokenizer=tokenizer)
    train_dataset = GraphTextDatasetFineTune(root='../data/', gt=gt, split='train',parameters= parameters, tokenizer=tokenizer)
    
    test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_epochs = parameters['nb_epochs']
    batch_size = parameters['batch_size']

    print_parameters(parameters)


    if parameters['model_name'] == "allenai/scibert_scivocab_uncased":
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_cids_dataset, batch_size=32, shuffle=False)
        test_text_loader = TorchDataLoader(test_text_dataset, batch_size=32, shuffle=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_cids_dataset, batch_size=100, shuffle=False)
        test_text_loader = TorchDataLoader(test_text_dataset, batch_size=100, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model_path = parameters['model_path']
    spec = importlib.util.spec_from_file_location("ModelModule", model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["ModelModule"] = model_module
    spec.loader.exec_module(model_module)

    # Import the right path 
    # Model = getattr(model_module, 'Model')
    model = Model(parameters)
    model.to(device)
    
    weight_decay = parameters['weight_decay']
    learning_rate = parameters['learning_rate']
    
    if parameters.get("use_SGD", False):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif parameters.get("load_LR_model", False):
        text_encoder_params = model.text_encoder.parameters()
        graph_encoder_params = model.graph_encoder.parameters()
        text_encoder_lr = parameters.get("text_encoder_lr", 1e-06)
        optimizer = optim.AdamW([
                    {'params': graph_encoder_params, 'lr': learning_rate, 'weight_decay': weight_decay},
                    {'params': text_encoder_params, 'lr': text_encoder_lr, 'weight_decay': weight_decay}
                ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                        betas=( parameters.get('beta1', 0.9), parameters.get('beta2', 0.999)),
                                        weight_decay=weight_decay)
    

    # print("Poids avant le chargement :")
    # for name, param in model.named_parameters():
    #     print(name)
    
    
    # print(model.text_encoder.bert.embeddings.LayerNorm.weight)

    if parameters['load_model_path'] != "None":
        try : 
            checkpoint = torch.load("pt/"+parameters['load_model_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            print("=== WE SUCCESSFULLY LOADED THE WEIGHTS OF A PREVIOUS MODEL ===")
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print("Error loading model or optimizer :", e)
            raise
    # print("\nPoids après le chargement :")
    # print(model.text_encoder.bert.embeddings.LayerNorm.weight)
    print("==========================================================", end='\n\n')
    print('Start training')
    
    model = model.get_graph_encoder()

    best_lraps = train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, test_loader, test_text_loader, printEvery = parameters.get("print_every", 1))

    return best_lraps

def train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, test_loader, test_text_loader, printEvery = -1):
    log_dir = parameters['log_dir']
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{log_dir}/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000
    lraps = 0
    
    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    cmt = 1
        
    for epoch in range(nb_epochs):
        
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        count_iter = 0
        for batch in train_loader:
            try:
                x_text = batch.text_embedding.to(device)
                x_text = batch.text_embedding.view(-1, parameters['nout']).to(device)
                graph_batch = batch
                graph_batch.pop('text_embedding')  # Remove the text embedding from the graph batch

                x_graph = model(graph_batch.to(device))
                
                if parameters["loss"] == "contrastive":
                    current_loss = contrastive_loss(x_graph, x_text)
                elif parameters["loss"] == "triplet":
                    current_loss = triplet_loss_sim(x_graph, x_text, parameters['margin_delta'])
                elif parameters["loss"] == "triplet and contrastive":
                    current_loss = triplet_loss_sim(x_graph, x_text, parameters['margin_delta']) + parameters['lambda_contra'] * parameters['lambda_param'] * contrastive_loss(x_graph, x_text)
                elif parameters["loss"] == "interpolate":
                    t = min(cmt/parameters['nb_epochs']/2,parameters['t_max'])
                    current_loss = triplet_loss_sim(x_graph, x_text, parameters['margin_delta']) * t + (1-t) * parameters['lambda_contra'] * parameters['lambda_param'] * contrastive_loss(x_graph, x_text)
                elif parameters["loss"] == "cosin":
                    current_loss = cosine_similarity_loss(x_graph, x_text)
                else:
                    current_loss = contrastive_loss(x_graph, x_text)
                if parameters['VQ']:
                    current_loss += model.quantization_loss * parameters.get("lambda_vq", 1e-02)
                    
                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                loss += current_loss.item()
                
                count_iter += 1
                
                if count_iter % printEvery == 0 and printEvery != -1:
                    time2 = time.time()
                    print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                                time2 - time1, loss/count_iter))
                torch.cuda.empty_cache() # test to liberate memory space
            except Exception as e:
                torch.cuda.empty_cache()
                print('An error occurred during training:', e)

        writer.add_scalar('Loss/train', loss, epoch)
        
        loss = 0
        model.eval()
        torch.cuda.empty_cache()
        
        val_loss = 0        
        for batch in val_loader:
            try:
                x_text = batch.text_embedding.to(device)
                x_text = batch.text_embedding.view(-1, parameters['nout']).to(device)
                graph_batch = batch
                graph_batch.pop('text_embedding')  # Remove the text embedding from the graph batch

                x_graph = model(graph_batch.to(device))
                
                if parameters["loss"] == "contrastive":
                    current_loss = contrastive_loss(x_graph, x_text)
                elif parameters["loss"] == "triplet":
                    current_loss = triplet_loss_sim(x_graph, x_text, parameters['margin_delta'])
                elif parameters["loss"] == "triplet and contrastive":
                    current_loss = triplet_loss_sim(x_graph, x_text, parameters['margin_delta']) + parameters['lambda_contra'] * parameters['lambda_param'] * contrastive_loss(x_graph, x_text)
                elif parameters["loss"] == "interpolate":
                    t = min(cmt/parameters['nb_epochs']/2,parameters['t_max'])
                    current_loss = triplet_loss_sim(x_graph, x_text, parameters['margin_delta']) * t + (1-t) * parameters['lambda_contra'] * parameters['lambda_param'] * contrastive_loss(x_graph, x_text)
                elif parameters["loss"] == "cosin":
                    current_loss = cosine_similarity_loss(x_graph, x_text)
                else:
                    current_loss = contrastive_loss(x_graph, x_text)
                if parameters['VQ']:
                    current_loss += model.quantization_loss * parameters.get("lambda_vq", 1e-02)
                    
                val_loss += current_loss.item()
                torch.cuda.empty_cache() # test to liberate memory space
            except Exception as e:
                torch.cuda.empty_cache()
                print('An error occurred during validation:', e)
        try:
            lraps = calculate_val_lraps_text(model, val_dataset, val_loader, device, parameters)
        except Exception as e:
            print('An error occurred during calculate_val_lraps:', e)
        torch.cuda.empty_cache() # test to liberate memory space
        best_validation_loss = min(best_validation_loss, val_loss)
        best_lraps = max(best_lraps, lraps)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        time2 = time.time()
        if best_lraps==lraps:
            print('lraps improoved saving checkpoint...')
            save_path = parameters.get("save_path", "model_checkpoint")
            save_path = os.path.join('pt', f"{save_path}_graph")
            print('-----EPOCH'+str(epoch+1)+'----- done')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, save_path)
            # try:
            #     make_csv_online(model, test_loader, test_text_loader, device, name=parameters['expt_name'])
            # except Exception as e:
            #     print('Could not generate the csv file because of the error :', e)
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps, "best current LRAPS is :",  best_lraps, "time : ", time2 - time1)
        else :
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps, "best current LRAPS is :",  best_lraps, "time : ", time2 - time1)
        torch.cuda.empty_cache() # test to liberate memory space
        
    return best_lraps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <file_path>")
        sys.exit(1)

    # Le premier argument après le nom du script est le file_path
    file_path = sys.argv[1]

    list_config_path = [file_path]
    train(list_config_path)