from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score
import pandas as pd

import os


def calculate_val_lraps_AMAN(model, discriminator, val_dataset, val_loader, device, save=False):
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
    
    top_k_indices = np.argsort(-similarity, axis=1)[:, :10]  # Get top 10 indices for each item

    adjusted_similarity = np.copy(similarity)

    for i in range(similarity.shape[0]):  # Iterate over each item
        for j in range(10):  # Iterate over top 10 closest points
            index = top_k_indices[i, j]
            
            # Create pairs for discriminator evaluation
            text_emb = text_embeddings_flat[i]
            graph_emb = graph_embeddings_flat[index]

            # Convert to tensor and pass through discriminator
            discriminator_score = discriminator(torch.tensor(text_emb), torch.tensor(graph_emb)).item()
            
            # Adjust the similarity score
            adjusted_similarity[i, index] -= discriminator_score

    # Calcul du LRAPS
    lrap_score = label_ranking_average_precision_score(true_labels, similarity)
    
    if save:
        solution = pd.DataFrame(similarity)
        solution['ID'] = solution.index
        solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]

        # Create 'submissions' folder if it doesn't exist
        if not os.path.exists('submissions'):
            os.makedirs('submissions')

        # Format the filename based on the LRAP score
        formatted_score = int(lrap_score * 10000)
        filename = f"{formatted_score}_submission.csv"

        # Save the DataFrame to the file in 'submissions' folder
        solution.to_csv(os.path.join('submissions', filename), index=False)

    return lrap_score

def calculate_val_lraps(model, val_dataset, val_loader, device, save=False):
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
    
    if save:
        solution = pd.DataFrame(similarity)
        solution['ID'] = solution.index
        solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]

        # Create 'submissions' folder if it doesn't exist
        if not os.path.exists('submissions'):
            os.makedirs('submissions')

        # Format the filename based on the LRAP score
        formatted_score = int(lrap_score * 10000)
        filename = f"{formatted_score}_submission.csv"

        # Save the DataFrame to the file in 'submissions' folder
        solution.to_csv(os.path.join('submissions', filename), index=False)

    return lrap_score

def calculate_val_lraps_VQ(model, val_dataset, val_loader, device, save = False):
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
    
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    if save:
        solution = pd.DataFrame(similarity)
        solution['ID'] = solution.index
        solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]

        # Create 'submissions' folder if it doesn't exist
        if not os.path.exists('submissions'):
            os.makedirs('submissions')

        # Format the filename based on the LRAP score
        formatted_score = int(lrap_score * 10000)
        filename = f"{formatted_score}_submission.csv"

        # Save the DataFrame to the file in 'submissions' folder
        solution.to_csv(os.path.join('submissions', filename), index=False)

    return lrap_score


