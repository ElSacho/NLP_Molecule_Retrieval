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

from models.disciminator import Discriminator
from losses import compute_triplet_loss, wgan_gp_loss, triplet_loss_sim
from torchvision import datasets, transforms

import json

from utils import calculate_val_lraps, calculate_val_lraps_VQ

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

BCEL = torch.nn.BCEWithLogitsLoss()

def negative_sampling_contrastive_loss(v1, v2, labels):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye), logits.diag() > 0

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
            print("weight loaded")
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print("Erreur lors du chargement du modèle ou de l'optimiseur :", e)
            raise
    
    print('Start training')
    if parameters.get('AMAN_freeze', False):
        print("you are using a discriminator with freeze decay")
        discriminator = Discriminator(parameters)
        discriminator_optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999),
                                    weight_decay=weight_decay)
        if parameters['load_model_path'] != "None":
            checkpoint_w = torch.load("discriminator_checkpoint.pt")
            discriminator.load_state_dict(checkpoint_w['model_state_dict'])
        best_lraps = train_after_loading_AMAN_freeze_decay(model, discriminator, optimizer, discriminator_optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, printEvery = parameters.get("print_every", 1))
    elif parameters.get('use_discriminator', False):
        print("you are using a discriminator")
        discriminator = Discriminator(parameters)
        discriminator_optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999),
                                    weight_decay=weight_decay)
        if parameters['load_model_path'] != "None":
            checkpoint_w = torch.load("discriminator_checkpoint.pt")
            discriminator.load_state_dict(checkpoint_w['model_state_dict'])
        best_lraps = train_after_loading_AMAN(model, discriminator, optimizer, discriminator_optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = parameters.get("print_every", 1))
    elif parameters.get('epochs_before_freeze', -1) != -1:
        print(1)
        best_lraps = train_after_loading_VQ_epochs_break(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, printEvery = parameters.get("print_every", 1))
    elif parameters.get('VQ', False):
        print(2)
        best_lraps = train_after_loading_VQ(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = parameters.get("print_every", 1))
    elif parameters.get('accumulation_step', 1) != 1:
        print(3)
        best_lraps = train_after_loading_accumulation(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = parameters.get("print_every", 1))
    else:
        print(4)
        best_lraps = train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = parameters.get("print_every", 1))
    
    return best_lraps

def train_after_loading(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = 1):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000
    
    # optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9)

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        count_iter = 0
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
            # has_nan = False
            # for name, module in model.named_modules():
            #     for param_name, param in module.named_parameters():
            #         if torch.isnan(param).any():
            #             print(f"NaN trouvé dans le module {name}, paramètre: {param_name}")
            #             has_nan = True
            #             break
            #     if has_nan:
            #         break
            # print(has_nan)
            loss += current_loss.item()
            # print(loss)
            
            
            count_iter += 1
            
            if count_iter % printEvery == 0 and printEvery != 1:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/count_iter))

            if count_iter > parameters['n_batches_before_break_in_epochs'] and parameters['n_batches_before_break_in_epochs'] != -1:
                break
            torch.cuda.empty_cache() # test to liberate memory space
        loss = 0
        writer.add_scalar('Loss/train', loss, epoch)
        
        model.eval()
        torch.cuda.empty_cache()
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
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        if best_lraps==lraps:
            print('lraps loss improoved saving checkpoint...')
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation improved loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
        else :
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
        torch.cuda.empty_cache() # test to liberate memory space
        

        
    return best_lraps

def train_after_loading_accumulation(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = 50):
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
    
    accumulation_steps = parameters['accumulation_step']  # Nombre d'itérations pour accumuler les gradients

    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        
        x_graph_accumulated = []
        count_iter = 0
        x_text_accumulated = []
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids') # This is likely done to prevent the input_ids from being processed in the subsequent graph operations, as they might be handled separately.
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            
            x_graph_accumulated.append(x_graph)
            with torch.no_grad():  # Gradients for x_text are not computed
                x_text_accumulated.append(x_text)
            
            if (count_iter + 1) % accumulation_steps == 0:
                # Concaténer les sorties accumulées
                x_graph_concat = torch.cat(x_graph_accumulated, dim=0)
                with torch.no_grad():  # Ensure no gradients are computed for x_text
                    x_text_concat = torch.cat(x_text_accumulated, dim=0)

                # Calculer la perte sur les sorties accumulées
                current_loss = contrastive_loss(x_graph_concat, x_text_concat) / accumulation_steps
                loss += current_loss.item()

                # Rétropropagation et mise à jour des poids
                current_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Réinitialiser les buffers
                x_graph_accumulated = []
                x_text_accumulated = []
            
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
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
        
        torch.cuda.empty_cache() # test to liberate memory space
    return best_lraps

def train_after_loading_VQ_epochs_break(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, printEvery = 1):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0
    cur_loss = 0
    vq_loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        # print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        if epoch == parameters['epochs_before_freeze']:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            model.text_encoder.train_mode(False)
        count_iter = 0
        for batch in train_loader:
            try :
                input_ids = batch.input_ids
                batch.pop('input_ids')
                attention_mask = batch.attention_mask
                batch.pop('attention_mask')
                graph_batch = batch

                # Move data to device (e.g., GPU)
                graph_batch = graph_batch.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Forward pass through the model
                quantized_graph, quantized_text, quantization_loss_graph, quantization_loss_text = model(graph_batch, input_ids, attention_mask)

                # Calculate Contrastive Loss
                current_loss = contrastive_loss(quantized_graph, quantized_text)

                # Include Quantization Loss
                # You can adjust the weight of these losses if necessary
                lambda_quantization = parameters['lambda_quantization']  # Example weight for quantization loss
                # total_loss = current_loss
                total_loss = current_loss + lambda_quantization * (quantization_loss_graph + quantization_loss_text)
                # total_loss = total_loss.mean()
                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accumulate the losses for logging or averaging
                loss += total_loss.item()
                cur_loss += current_loss.item()
                vq_loss += (lambda_quantization * (quantization_loss_graph + quantization_loss_text)).item()
                count_iter += 1
            
                if count_iter % printEvery == 0 and printEvery!= 1:
                    time2 = time.time()
                    print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                                time2 - time1, loss/count_iter))
                    print("And the loss decompose in : ", cur_loss/count_iter, 'and ', vq_loss/count_iter)

                if count_iter > parameters['n_batches_before_break_in_epochs'] and parameters['n_batches_before_break_in_epochs'] != -1:
                    break
                torch.cuda.empty_cache() # test to liberate memory space
            except:
                pass
        loss = 0
        cur_loss = 0
        vq_loss = 0
        writer.add_scalar('Loss/train', loss, epoch)
        
        model.eval()
        val_loss = 0        
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            # Move data to device
            graph_batch = graph_batch.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            quantized_graph, quantized_text, quantization_loss_graph, quantization_loss_text = model(graph_batch, input_ids, attention_mask)

            # Calculate Contrastive Loss (primary metric for validation)
            current_loss = contrastive_loss(quantized_graph, quantized_text)

            # Optionally, you can track quantization loss for analysis
            # However, it's not used for updating model parameters during validation
            val_loss += current_loss.item()
            count_iter += 1
            torch.cuda.empty_cache() # test to liberate memory space
        lraps = calculate_val_lraps_VQ(model, val_dataset, val_loader, device)
        torch.cuda.empty_cache() # test to liberate memory space
        best_validation_loss = min(best_validation_loss, val_loss)
        best_lraps = max(best_lraps, lraps)
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        if best_lraps==lraps:
            time2 = time.time()
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS IMPROVED:', lraps, "time : ", time2 - time1, ' s')
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
        else:
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
        
        torch.cuda.empty_cache() # test to liberate memory space
    return best_lraps

def train_after_loading_discriminator(model, discriminator, optimizer, discriminator_optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = 1):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000
    
    # optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9)
    
    lambda_param = parameters['lambda_param']
    margin_delta = parameters['margin_delta']

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        count_iter = 0
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids') # This is likely done to prevent the input_ids from being processed in the subsequent graph operations, as they might be handled separately.
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            
            # triplet_loss = compute_triplet_loss(x_graph, x_text, discriminator, margin_delta)
            # Pour le discriminateur
            adv_loss = wgan_gp_loss(discriminator, x_graph, x_text, parameters['lambda_gp'])
            # Pour le modèle principal
            current_loss = contrastive_loss(x_graph, x_text)
            total_loss = lambda_param * adv_loss + current_loss
            optimizer.zero_grad()  # Réinitialisez les gradients du modèle principal
            total_loss.backward(retain_graph=True)  # Rétropropagation pour total_loss, en conservant le graphe
            optimizer.step()  # Mise à jour des poids du modèle principal

            # Pour le discriminateur
            discriminator_optimizer.zero_grad()  # Réinitialisez les gradients du discriminateur
            adv_loss.backward()  # Rétropropagation uniquement pour adv_loss
            discriminator_optimizer.step()  # Mise à jour des poids du discriminateur

            
            loss += total_loss.item()
            
            count_iter += 1
            
            if count_iter % printEvery == 0 and printEvery != 1:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/count_iter))
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

            # Compute adversarial loss
            # Pour le discriminateur
            # Pour le modèle principal
            adv_loss = wgan_gp_loss(discriminator, x_graph, x_text, parameters['lambda_gp'])
            current_loss = contrastive_loss(x_graph, x_text)
            total_loss = lambda_param * adv_loss + current_loss
            val_loss += total_loss.item()
            torch.cuda.empty_cache() # test to liberate memory space
        lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
        torch.cuda.empty_cache() # test to liberate memory space
        best_validation_loss = min(best_validation_loss, val_loss)
        best_lraps = max(best_lraps, lraps)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        if best_lraps==lraps:
            # print('lraps loss improoved saving checkpoint...')
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation improved loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
            torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'discriminator_checkpoint.pt')
        else :
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
        torch.cuda.empty_cache() # test to liberate memory space
        
    return best_lraps

def train_after_loading_AMAN_freeze_decay(model, discriminator, optimizer, discriminator_optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, printEvery = 1):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000
    
    # optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9)
    
    lambda_param = parameters['lambda_param']
    margin_delta = parameters['margin_delta']

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    cmt = 1
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        if epoch == cmt*parameters['epochs_decay']:
            try:
                model.freeze_layers(cmt)
                train_loader = DataLoader(train_dataset, batch_size = parameters['batch_size'] + parameters['batch_size_add']*cmt, shuffle=True)
                # val_loader = DataLoader(val_dataset, batch_size = parameters['batch_size'] + parameters['batch_size_add']*cmt, shuffle=True)
                print('Freezed', cmt,'layers and batch size of :',parameters['batch_size'] + parameters['batch_size_add']*cmt )
            except:
                print("The model is fully freezed")
            cmt += 1
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        count_iter = 0
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids') # This is likely done to prevent the input_ids from being processed in the subsequent graph operations, as they might be handled separately.
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            
            triplet_loss = triplet_loss_sim(x_graph, x_text, margin_delta)
            current_loss = contrastive_loss(x_graph, x_text)
            # Compute adversarial loss
            adv_loss = wgan_gp_loss(discriminator, x_graph, x_text, parameters['lambda_gp'])

            # Combined loss
            total_loss = triplet_loss + lambda_param * (adv_loss * 0.2 + current_loss)
            # print("losses triplet :",triplet_loss.item(), " adv :", adv_loss.item(), "contras :", current_loss.item())
            
            # Zero gradients for optimizer and discriminator optimizer
            optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # Backward pass and optimizers step
            total_loss.backward()
            optimizer.step()
            discriminator_optimizer.step()
            
            loss += total_loss.item()
            
            count_iter += 1
            
            if count_iter % printEvery == 0 and printEvery != 1:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/count_iter))
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
            
            triplet_loss = triplet_loss_sim(x_graph, x_text, margin_delta)

            # Compute adversarial loss
            adv_loss = wgan_gp_loss(discriminator, x_graph, x_text, parameters['lambda_gp'])
            current_loss = contrastive_loss(x_graph, x_text)

            # Combined loss
            total_loss = triplet_loss + lambda_param * (adv_loss * 0.2 + current_loss)
            val_loss += total_loss.item()
            torch.cuda.empty_cache() # test to liberate memory space
        lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
        torch.cuda.empty_cache() # test to liberate memory space
        best_validation_loss = min(best_validation_loss, val_loss)
        best_lraps = max(best_lraps, lraps)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        if best_lraps==lraps:
            # print('lraps loss improoved saving checkpoint...')
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation improved loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation improved loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
            torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'discriminator_checkpoint.pt')
            # print('checkpoint saved to: {}'.format(save_path))
            
        else :
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
        torch.cuda.empty_cache() # test to liberate memory space
        
    return best_lraps




def train_after_loading_AMAN(model, discriminator, optimizer, discriminator_optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = 1):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000
    
    # optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9)
    
    lambda_param = parameters['lambda_param']
    margin_delta = parameters['margin_delta']

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        count_iter = 0
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids') # This is likely done to prevent the input_ids from being processed in the subsequent graph operations, as they might be handled separately.
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            
            triplet_loss = triplet_loss_sim(x_graph, x_text, margin_delta)
            current_loss = contrastive_loss(x_graph, x_text)
            # Compute adversarial loss
            adv_loss = wgan_gp_loss(discriminator, x_graph, x_text, parameters['lambda_gp'])

            # Combined loss
            total_loss = triplet_loss + lambda_param * (adv_loss * 0.2 + current_loss)
            # print("losses triplet :",triplet_loss.item(), " adv :", adv_loss.item(), "contras :", current_loss.item())
            
            # Zero gradients for optimizer and discriminator optimizer
            optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # Backward pass and optimizers step
            total_loss.backward()
            optimizer.step()
            discriminator_optimizer.step()
            
            loss += total_loss.item()
            
            count_iter += 1
            
            if count_iter % printEvery == 0 and printEvery != 1:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/count_iter))
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
            
            triplet_loss = triplet_loss_sim(x_graph, x_text, margin_delta)

            # Compute adversarial loss
            adv_loss = wgan_gp_loss(discriminator, x_graph, x_text, parameters['lambda_gp'])
            current_loss = contrastive_loss(x_graph, x_text)

            # Combined loss
            total_loss = triplet_loss + lambda_param * (adv_loss * 0.2 + current_loss)
            
            # Zero gradients for optimizer and discriminator optimizer
            optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # Backward pass and optimizers step
            total_loss.backward()
            optimizer.step()
            discriminator_optimizer.step()
            
            val_loss += total_loss.item()
            torch.cuda.empty_cache() # test to liberate memory space
        lraps = calculate_val_lraps(model, val_dataset, val_loader, device)
        torch.cuda.empty_cache() # test to liberate memory space
        best_validation_loss = min(best_validation_loss, val_loss)
        best_lraps = max(best_lraps, lraps)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Lraps/val', lraps, epoch)
        if best_lraps==lraps:
            # print('lraps loss improoved saving checkpoint...')
            save_path = os.path.join('./models', 'model_attention'+str(epoch)+'.pt')
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation improved loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
        else :
            print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)), 'and LRAPS :', lraps )
        torch.cuda.empty_cache() # test to liberate memory space
        
    return best_lraps

def train_after_loading_VQ(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, printEvery = 50):
    expt_name = parameters['expt_name']
    timestamp = time.strftime("%Y-%m-%d--%H%M")
    writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    loss = 0
    cur_loss = 0
    vq_loss = 0

    count_iter = 0
    time1 = time.time()

    best_validation_loss = 1_000_000

    writer.add_hparams(hparam_dict=parameters, metric_dict={})
    torch.cuda.empty_cache() # test to liberate memory space
    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()
        count_iter = 0
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            # Move data to device (e.g., GPU)
            graph_batch = graph_batch.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass through the model
            quantized_graph, quantized_text, quantization_loss_graph, quantization_loss_text = model(graph_batch, input_ids, attention_mask)

            # Calculate Contrastive Loss
            current_loss = contrastive_loss(quantized_graph, quantized_text)

            # Include Quantization Loss
            # You can adjust the weight of these losses if necessary
            lambda_quantization = parameters['lambda_quantization']  # Example weight for quantization loss
            # total_loss = current_loss
            total_loss = current_loss + lambda_quantization * (quantization_loss_graph + quantization_loss_text)
            # total_loss = total_loss.mean()
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate the losses for logging or averaging
            loss += total_loss.item()
            cur_loss += current_loss.item()
            vq_loss += (lambda_quantization * (quantization_loss_graph + quantization_loss_text)).item()
            count_iter += 1
            
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/count_iter))
                print("And the loss decompose in : ", cur_loss/count_iter, 'and ', vq_loss/count_iter)

            if count_iter > parameters['n_batches_before_break_in_epochs'] and parameters['n_batches_before_break_in_epochs'] != -1:
                break
            torch.cuda.empty_cache() # test to liberate memory space
        loss = 0
        cur_loss = 0
        vq_loss = 0
        writer.add_scalar('Loss/train', loss, epoch)
        
        model.eval()
        val_loss = 0        
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            # Move data to device
            graph_batch = graph_batch.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            quantized_graph, quantized_text, quantization_loss_graph, quantization_loss_text = model(graph_batch, input_ids, attention_mask)

            # Calculate Contrastive Loss (primary metric for validation)
            current_loss = contrastive_loss(quantized_graph, quantized_text)

            # Optionally, you can track quantization loss for analysis
            # However, it's not used for updating model parameters during validation
            val_loss += current_loss.item()
            count_iter += 1
            torch.cuda.empty_cache() # test to liberate memory space
        lraps = calculate_val_lraps_VQ(model, val_dataset, val_loader, device)
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
            # 'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, 'model_checkpoint.pt')
            print('checkpoint saved to: {}'.format(save_path))
        
        torch.cuda.empty_cache() # test to liberate memory space
    return best_lraps