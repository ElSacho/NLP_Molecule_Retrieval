from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score
import pandas as pd

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

