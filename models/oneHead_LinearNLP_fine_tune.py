from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel, BertModel

from models.quantization import QuantizationLayer

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, global_mean_pool, BatchNorm

class GraphEncoderOneHead(nn.Module):
    def __init__(self, parameters):            
        super(GraphEncoderOneHead, self).__init__()
        num_node_features = parameters['num_node_features']
        nout = parameters['nout']
        nhid = parameters['nhid']
        graph_hidden_channels = parameters['graph_hidden_channels']
        mlp_layers = parameters['mlp_layers']
        use_sage = parameters['use_sage']
        use_cheb = parameters['use_cheb']
        num_heads = parameters['num_head']
        dropout_rate = parameters['dropout_rate']
        self.cheb_k = 5
        self.temp = nn.Parameter(torch.Tensor([parameters['tempGraph']])) 
        self.num_heads = num_heads
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.ln = nn.LayerNorm(nout)
        self.dropout = nn.Dropout(dropout_rate)
        
        # GCN layers with BatchNorm
        if use_sage:
            self.conv1 = SAGEConv(num_node_features, graph_hidden_channels)
            self.conv2 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
        elif use_cheb:
            self.conv1 = ChebConv(num_node_features, graph_hidden_channels, self.cheb_k)
            self.conv2 = ChebConv(graph_hidden_channels, graph_hidden_channels, self.cheb_k)
            self.conv3 = ChebConv(graph_hidden_channels, graph_hidden_channels, self.cheb_k)
        else:
            self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
            self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
            
        self.bn1 = BatchNorm(graph_hidden_channels)
        self.bn2 = BatchNorm(graph_hidden_channels)
        self.bn3 = BatchNorm(graph_hidden_channels)
        
        # Attention layer with BatchNorm
        # self.attention = GATConv(graph_hidden_channels, graph_hidden_channels // num_heads, heads=num_heads)
        # self.bn_attention = BatchNorm(graph_hidden_channels)  # No change needed here
        self.attention = GATConv(graph_hidden_channels, graph_hidden_channels // num_heads, heads=num_heads)
        self.bn_attention = BatchNorm(graph_hidden_channels)  # Adjust for multi-head output
        
        # Linear layers
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hiddens = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _ in range(mlp_layers):
            self.mol_hiddens.append(nn.Linear(nhid, nhid).to(device))
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        
        # Apply GCN layers with residual connections
        x = self.relu(self.bn1(self.conv1(x, edge_index))) + x
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x, edge_index))) + x
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x, edge_index))) + x
        x = self.dropout(x)

        # Apply attention mechanism with residual connection
        if self.num_heads != 1:     
            attention_output = self.attention(x, edge_index)
            attention_output = self.bn_attention(attention_output)
            attention_output = self.relu(attention_output)
            x = attention_output  # Directly use the attention output
        else:
            x = self.relu(self.bn_attention(self.attention(x, edge_index))) + x

        # Pooling and linear layers
        x = global_mean_pool(x, batch)
        x = self.relu(self.mol_hidden1(x))
        
        if len(self.mol_hiddens) > 0:
            for mlp_layer in self.mol_hiddens:
                x = self.relu(mlp_layer(x))
                x = self.dropout(x)
        x = self.mol_hidden2(x)

        x = x * torch.exp(self.temp)

        return x
    
class TextEncoder(nn.Module):
    def __init__(self, parameters):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(parameters['model_name'])
        nout = parameters['nout']
        self.temp = nn.Parameter(torch.Tensor([parameters['tempText']])) 
        self.linear = nn.Linear(self.bert.config.hidden_size, nout)
        self.norm = nn.LayerNorm(nout)
        if parameters['fine_tune']:
            self.train_mode()

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        cls_token_state = encoded_text.last_hidden_state[:, 0, :]
        linear_output = self.linear(cls_token_state)
        normalized_output = self.norm(linear_output)
        text_x = normalized_output * torch.exp(self.temp)
        return text_x
    
    def train_mode(self, mode=True):
        self.fine_tune = mode
        for param in self.bert.parameters():
            param.requires_grad = mode

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoderOneHead(parameters)
        self.text_encoder = TextEncoder(parameters)
        self.vq = parameters['VQ']
        if self.vq :
            self.quantization = QuantizationLayer(parameters)
            
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        if self.vq :
            quantized_graph, quantization_loss_graph = self.quantization(graph_encoded)
            quantized_text, quantization_loss_text = self.quantization(text_encoded)
            return graph_encoded, text_encoded, quantization_loss_graph, quantization_loss_text
        
        # print(graph_encoded.mean())
        # print(text_encoded.mean())
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
