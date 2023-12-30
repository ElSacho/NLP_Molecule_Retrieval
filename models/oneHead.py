from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel, BertModel


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm

class GraphEncoderOneHead(nn.Module):
    def __init__(self, parameters):            
        super(GraphEncoderOneHead, self).__init__()
        num_node_features = parameters['num_node_features']
        nout = parameters['nout']
        nhid = parameters['nhid']
        graph_hidden_channels = parameters['graph_hidden_channels']
        mlp_layers = parameters['mlp_layers']
        num_heads = 1
        
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
        self.mol_hiddens = []
        for _ in range(mlp_layers):
            self.mol_hiddens.append(nn.Linear(nhid, nhid))
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
        if len(self.mol_hiddens) > 0:
            for mlp_layer in self.mol_hiddens:
                x = self.relu(mlp_layer(x))
        x = self.mol_hidden2(x)

        return x
    
class TextEncoder(nn.Module):
    def __init__(self, parameters):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(parameters['model_name'])

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:,0,:]

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoderOneHead(parameters)
        self.text_encoder = TextEncoder(parameters)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
