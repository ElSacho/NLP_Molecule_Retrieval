import torch
import torch.nn.functional as F
from tqdm import tqdm


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
from sklearn.cluster import KMeans
from models.oneHead_LinearNLP import Model

import matplotlib.pyplot as plt
# from models.disciminator import Discriminator
from losses import wgan_gp_loss, triplet_loss_sim, contrastive_loss, cosine_similarity_loss
from torchvision import datasets, transforms

import json

from utils import calculate_val_lraps, calculate_val_lraps_VQ, print_parameters
from generate_submission import make_csv_online


def clustering(list_config_path):
    best_lraps = 0
    for config_path in list_config_path:
        best_lraps = clustering_conf(config_path, best_lraps)

def clustering_conf(config_path, best_lraps):
    
    with open(config_path, 'r') as file:
        parameters = json.load(file)

    model_name = parameters['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='../data/', gt=gt, split='train', tokenizer=tokenizer)
    
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
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_cids_dataset, batch_size=32, shuffle=False)
        test_text_loader = TorchDataLoader(test_text_dataset, batch_size=32, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
    print('Start loading the Mixture')

    best_lraps = plot_KMeans(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, test_loader, test_text_loader, printEvery = parameters.get("print_every", 1))

    return best_lraps

def plot_KMeans(model, optimizer, nb_epochs, train_loader, val_loader, val_dataset, parameters, best_lraps, train_dataset, test_loader, test_text_loader, printEvery = -1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Compute Embeddings
    x_text_embeddings = []
    x_graph_embeddings = []
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for batch in tqdm(train_loader):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            graph_batch = batch
            graph_batch.pop('input_ids')
            graph_batch.pop('attention_mask')

            x_graph, x_text = model(graph_batch.to(device), input_ids, attention_mask)

            x_text_embeddings.append(x_text.cpu().numpy())
            x_graph_embeddings.append(x_graph.cpu().numpy())


    x_text_embeddings = np.concatenate(x_text_embeddings, axis=0)
    x_graph_embeddings = np.concatenate(x_graph_embeddings, axis=0)

    # Step 2: Perform K-Means Clustering
    x_graph_embeddings = x_graph_embeddings.astype(np.float32)

    # Step 2: Perform K-Means Clustering with explicit n_init
    kmeans = KMeans(n_clusters=10, n_init=10, random_state=0).fit(x_text_embeddings)
    labels = kmeans.labels_
    
    print("KMeans finished")

    # Step 3: Save x_graph Embeddings (already saved in x_graph_embeddings)

    # Step 4: Evaluate and Plot
    # Assuming each x_graph corresponds to an x_text, count mismatches
    print("Shape and type of an embedding:", x_graph_embeddings[0].shape, x_graph_embeddings[0].dtype)
    print("Shape and type of an text:", x_text_embeddings[0].shape, x_text_embeddings[0].dtype)

    mismatch_counts = np.zeros(10, dtype=int)
    for i, label in enumerate(labels):
        # Simplified prediction step
        assigned_cluster = kmeans.predict(x_graph_embeddings[i:i+1])[0]
        if assigned_cluster != label:
            mismatch_counts[label] += 1
            
    # Plotting
    plt.bar(range(10), mismatch_counts)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Mismatches')
    plt.title('Mismatches in x_graph per Cluster')
    plt.savefig(f"KMeans k={10}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <file_path>")
        sys.exit(1)

    # Le premier argument après le nom du script est le file_path
    file_path = sys.argv[1]

    list_config_path = [file_path]
    clustering(list_config_path)