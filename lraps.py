from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
from NLP_Molecule_Retrieval.ModelOneHead import Model
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



if __name__ == '__main__':
    # model_name = 'distilbert-base-uncased'
    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='data/', gt=gt, split='val', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 10

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
    model.to(device)

    model_path = "model_checkpoint.pt"
    print('loading best model...')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('loading suceed')
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

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import label_ranking_average_precision_score

    # similarity = cosine_similarity(text_embeddings, graph_embeddings)

    # Puisque la matrice de labels vrais est l'identité pour votre cas,
    # nous pouvons la créer facilement avec np.eye
    num_samples = len(val_dataset)
    true_labels = np.eye(num_samples)

    # Flatten les embeddings pour les aligner avec la forme attendue par cosine_similarity
    graph_embeddings_flat = [item for sublist in graph_embeddings for item in sublist]
    text_embeddings_flat = [item for sublist in text_embeddings for item in sublist]

    # Calcul de la similarité cosinus entre les embeddings
    similarity = cosine_similarity(text_embeddings_flat, graph_embeddings_flat)

    # Calcul du LRAPS
    lrap_score = label_ranking_average_precision_score(true_labels, similarity)

    print("Label Ranking Average Precision Score: ", lrap_score)
