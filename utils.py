from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
from dataloader import GraphTextDataset, GraphDataset, TextDataset
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score
import pandas as pd

import os

def calculate_val_lraps_text(model, val_dataset, val_loader, device, parameters):
    graph_embeddings = []
    text_embeddings = []

    for batch in val_loader:        
        x_text = batch.text_embedding.to(device)
        x_text = batch.text_embedding.view(-1, parameters['nout']).to(device)
        graph_batch = batch
        graph_batch.pop('text_embedding')  # Remove the text embedding from the graph batch

        x_graph = model(graph_batch.to(device))
        graph_embeddings.append(x_graph.tolist())
        text_embeddings.append(x_text.tolist())

    num_samples = len(val_dataset)
    true_labels = np.eye(num_samples)

    # Flatten les embeddings pour les aligner avec la forme attendue par cosine_similarity
    
    graph_embeddings_flat = [item for sublist in graph_embeddings for item in sublist]
    text_embeddings_flat = [item for sublist in text_embeddings for item in sublist]

    # Calcul de la similarité cosinus entre les embeddings
    
    similarity = cosine_similarity(text_embeddings_flat, graph_embeddings_flat)

    # Calcul du LRAPS
    
    lrap_score = label_ranking_average_precision_score(true_labels, similarity)
    
    return lrap_score


def calculate_val_lraps(model, val_dataset, val_loader, device):
    graph_embeddings = []
    text_embeddings = []

    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        graph_embeddings.append(x_graph.tolist())
        text_embeddings.append(x_text.tolist())

    num_samples = len(val_dataset)
    true_labels = np.eye(num_samples)

    # Flatten les embeddings pour les aligner avec la forme attendue par cosine_similarity
    
    graph_embeddings_flat = [item for sublist in graph_embeddings for item in sublist]
    text_embeddings_flat = [item for sublist in text_embeddings for item in sublist]

    # Calcul de la similarité cosinus entre les embeddings
    
    similarity = cosine_similarity(text_embeddings_flat, graph_embeddings_flat)

    # Calcul du LRAPS
    
    lrap_score = label_ranking_average_precision_score(true_labels, similarity)
    
    return lrap_score

def calculate_val_lraps_VQ(model, val_dataset, val_loader, device):
    graph_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            quantized_graph, quantized_text, _, _ = model(graph_batch.to(device), input_ids.to(device), attention_mask.to(device))

            graph_embeddings.append(quantized_graph.cpu().numpy())
            text_embeddings.append(quantized_text.cpu().numpy())

    # Concatenate embeddings instead of flattening, for efficiency
    graph_embeddings = np.concatenate(graph_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)

    num_samples = len(val_dataset)
    true_labels = np.eye(num_samples)

    # Calculate cosine similarity
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    # Compute LRAP score
    lrap_score = label_ranking_average_precision_score(true_labels, similarity)

    return lrap_score


def print_parameters(parameters):
    separator = "=" * 80
    title = " THE PARAMETERS YOU USE "
    
    print(f"\n\n{separator}")
    print(f"{title.center(80, '=')}")
    print(f"{separator}")
    
    print(f"{'MODEL:'.ljust(20)} {parameters['model_name']}")
    print(f"{'BATCH_SIZE:'.ljust(20)} {parameters['batch_size']}")
    print(f"{'LOSS:'.ljust(20)} {parameters['loss']}")
    freeze_layers = parameters.get('num_layers_to_freeze', 0)
    print(f"{'FREEZING:'.ljust(20)} {freeze_layers} LAYERS")
    print(f"{'USING VQ:'.ljust(20)} {parameters['VQ']}")
    print(f"{separator}\n")