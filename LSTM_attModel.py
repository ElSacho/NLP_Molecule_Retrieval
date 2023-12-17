from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm

class GraphEncoderAttentionWithNeighborAttentionResidual(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, num_heads=1):
        super(GraphEncoderAttentionWithNeighborAttentionResidual, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(nout)
        
        # GCN layers with BatchNorm
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.bn1 = BatchNorm(graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.bn2 = BatchNorm(graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.bn3 = BatchNorm(graph_hidden_channels)
        
        # Attention layer with BatchNorm
        self.attention = GATConv(graph_hidden_channels, graph_hidden_channels, heads=num_heads)
        self.bn_attention = BatchNorm(graph_hidden_channels)
        
        # Linear layers
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        
        # Apply GCN layers with residual connections
        x = self.relu(self.bn1(self.conv1(x, edge_index))) + x
        x = self.relu(self.bn2(self.conv2(x, edge_index))) + x
        x = self.relu(self.bn3(self.conv3(x, edge_index))) + x

        # Apply attention mechanism with residual connection
        x = self.relu(self.bn_attention(self.attention(x, edge_index))) + x

        # Pooling and linear layers
        x = global_mean_pool(x, batch)
        x = self.relu(self.mol_hidden1(x))
        x = self.mol_hidden2(x)

        return x
    
class ChemLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, output_dim = 256, bidirectional=True):
        super(ChemLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # Prendre l'état caché final pour chaque séquence
        final_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        linear_output = self.linear(final_state)
        normalized_output = self.norm(linear_output)
        return normalized_output
    
class Model(nn.Module):
    def __init__(self, vocab_size, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoderAttentionWithNeighborAttentionResidual(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = ChemLSTMEncoder(vocab_size, output_dim = nout)
        
    def forward(self, graph_batch, input_seqs, input_lengths):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_seqs, input_lengths)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
class SmilesTokenizer:
    def __init__(self, smiles_list):
        self.char_to_idx = self._create_vocab(smiles_list)
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def _create_vocab(self, smiles_list):
        vocab = set()
        for smiles in smiles_list:
            vocab.update(set(smiles))
        vocab = sorted(list(vocab))
        return {char: idx for idx, char in enumerate(vocab, start=1)}  # Start indexing from 1

    def encode(self, smiles):
        return [self.char_to_idx[char] for char in smiles]

    def decode(self, token_ids):
        return ''.join([self.idx_to_char[idx] for idx in token_ids if idx in self.idx_to_char])
