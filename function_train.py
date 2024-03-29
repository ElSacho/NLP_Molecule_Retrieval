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

from models.oneHead_LinearNLP import Model

# from models.disciminator import Discriminator
from losses import wgan_gp_loss, triplet_loss_sim, contrastive_loss, cosine_similarity_loss,lifted_structured_loss
from torchvision import datasets, transforms

import json

from utils import calculate_val_lraps, calculate_val_lraps_VQ, print_parameters
from generate_submission import make_csv_online


def train(list_config_path):
    best_lraps = 0
    for config_path in list_config_path:
        best_lraps = train_conf(config_path, best_lraps)

def train_conf(config_path, best_lraps):
    
    with open(config_path, 'r') as file:
        parameters = json.load(file)

    model_name = parameters['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='../data/', gt=gt, split='train', tokenizer=tokenizer)
    
    test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)
    
    val_cids_dataset = GraphDataset(root='../data/', gt=gt, split='val_cids')
    val_text_dataset = TextDataset(file_path='../data/val_text.txt', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_epochs = parameters['nb_epochs']
    batch_size = parameters['batch_size']

    print_parameters(parameters)


    if parameters['model_name'] == "allenai/scibert_scivocab_uncased":
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_cids_dataset, batch_size=32, shuffle=False)
        val_csv_loader = DataLoader(val_cids_dataset, batch_size=32, shuffle=False)
        val_text_loader = TorchDataLoader(val_text_dataset, batch_size=32, shuffle=False)
        test_text_loader = TorchDataLoader(test_text_dataset, batch_size=32, shuffle=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_cids_dataset, batch_size=100, shuffle=False)
        val_csv_loader = DataLoader(val_cids_dataset, batch_size=100, shuffle=False)
        val_text_loader = TorchDataLoader(val_text_dataset, batch_size=100, shuffle=False)
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




    best_lraps = train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, test_loader, test_text_loader, val_csv_loader, val_text_loader, printEvery = parameters.get("print_every", 1))

    return best_lraps

def train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, test_loader, test_text_loader, val_csv_loader, val_text_loader, printEvery = -1):
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
    use_mutation = parameters.get("mutation", False)
    mutate_cmt = 0
    
    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    cmt = 1
        
    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        if use_mutation:
            if epoch % parameters["mutation_epochs"] == 0:
                if epoch >1:
                    try:
                        csv_name = parameters['expt_name'] + str(mutate_cmt)
                        make_csv_online(model, test_loader, test_text_loader, device, name=csv_name)
                    except Exception as e:
                        print('Could not generate the csv file because of the error :', e)
                mutate_cmt += 1
                print("mutation with the mutation rate : ", parameters["mutation_rate"])
                model.mutate(parameters["mutation_rate"])
                try:
                    before_mutate_lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
                    print("The LRAPS before mutation is : ", before_mutate_lraps)
                except Exception as e:
                    torch.cuda.empty_cache()
                    print('An error occurred during LRAPS calculation:', e)
                parameters["mutation_rate"] -= 1e-7
        model.train()
        count_iter = 0
        for batch in train_loader:
            try:
                if epoch == cmt * parameters.get('epochs_decay', -1):
                    if model.max_layers_to_freeze >= cmt:
                        model.freeze_layers(cmt)
                        train_loader = DataLoader(train_dataset, batch_size = parameters['batch_size'] + parameters['batch_size_add']*cmt, shuffle=True)
                        print('Freezed ', cmt,' layers and batch size of :', parameters['batch_size'] + parameters['batch_size_add']*cmt )
                    else:
                        print("The model is fully freezed")
                    cmt += 1
                    
                input_ids = batch.input_ids
                batch.pop('input_ids') # This is likely done to prevent the input_ids from being processed in the subsequent graph operations, as they might be handled separately.
                attention_mask = batch.attention_mask
                batch.pop('attention_mask')
                graph_batch = batch
                
                x_graph, x_text = model(graph_batch.to(device), 
                                        input_ids.to(device), 
                                        attention_mask.to(device))
                
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
                elif  parameters["loss"] == "lifted_structured_loss":
                    current_loss = lifted_structured_loss(x_graph, x_text, parameters['margin_delta'])
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
                input_ids = batch.input_ids
                batch.pop('input_ids')
                attention_mask = batch.attention_mask
                batch.pop('attention_mask')
                graph_batch = batch
                
                x_graph, x_text = model(graph_batch.to(device), 
                                        input_ids.to(device), 
                                        attention_mask.to(device))
                
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
                elif  parameters["loss"] == "lifted_structured_loss":
                    current_loss = lifted_structured_loss(x_graph, x_text, parameters['margin_delta'])
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
            lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
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
            save_path = parameters.get("save_path", "model_checkpoint.pt")
            # if use_mutation :
            #     save_path += str(mutate_cmt)
            save_path = os.path.join('pt', save_path)
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation improved loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps, "time : ", time2 - time1)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, save_path)
            try:
                csv_name = parameters['expt_name']
                if use_mutation:
                    csv_name += str(mutate_cmt)
                make_csv_online(model, test_loader, test_text_loader, device, name=csv_name)
                make_csv_online(model, val_csv_loader, val_text_loader, device, name='val_'+csv_name)
            except Exception as e:
                print('Could not generate the csv file because of the error :', e)
        else :
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps, "best current LRAPS is :",  best_lraps, "time : ", time2 - time1)
        torch.cuda.empty_cache() # test to liberate memory space
        
    return best_lraps
