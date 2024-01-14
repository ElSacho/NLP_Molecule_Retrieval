from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel, BertModel

from models.quantization import QuantizationLayer

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _ in range(mlp_layers):
            self.mol_hiddens.append(nn.Linear(nhid, nhid).to(device))
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
        nout = parameters['nout']
        self.linear = nn.Linear(self.bert.config.hidden_size, nout)
        self.norm = nn.LayerNorm(nout)
        num_layers_to_freeze = parameters.get('num_layers_to_freeze', 0)
        if parameters['model_name'] == "allenai/scibert_scivocab_uncased":
            self.max_number_to_freeze_scibert()
        else:
            self.max_number_to_freeze()
        if num_layers_to_freeze != 0:
            if parameters['model_name'] == "allenai/scibert_scivocab_uncased":
                self.freeze_layers_scibert(num_layers_to_freeze)
            else:
                self.freeze_layers(num_layers_to_freeze)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        cls_token_state = encoded_text.last_hidden_state[:, 0, :]
        linear_output = self.linear(cls_token_state)
        normalized_output = self.norm(linear_output)
        return normalized_output
    
    def max_number_to_freeze(self):
        max_layers_to_freeze = 0
        for _ in self.bert.transformer.layer:
            max_layers_to_freeze += 1
        self.max_layers_to_freeze = max_layers_to_freeze
        print("The max number of freezable layers is ,",max_layers_to_freeze)
        
    def max_number_to_freeze_scibert(self):
        max_layers_to_freeze = 0
        for layer in self.bert.encoder.layer:
            max_layers_to_freeze += 1
        self.max_layers_to_freeze = max_layers_to_freeze
        print("The max number of freezable layers is ,",max_layers_to_freeze)
    
    def freeze_layers(self, num_layers_to_freeze):
            # Compteur pour les couches gelées
        frozen_layers = 0

        # Parcourir les couches du transformer
        for layer in self.bert.transformer.layer:
            if frozen_layers < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_layers += 1
            else:
                break
    
    def freeze_layers_scibert(self, num_layers_to_freeze):
        # Freeze the first 'num_layers_to_freeze' layers
        print("freezing")
        for layer in self.bert.encoder.layer[:num_layers_to_freeze]:
            print(layer)
            for param in layer.parameters():
                param.requires_grad = False

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoderOneHead(parameters)
        self.text_encoder = TextEncoder(parameters)
        self.param = parameters
        print(self.text_encoder)
        self.vq = parameters['VQ']
        if self.vq :
            self.quantization = QuantizationLayer(parameters)
        self.max_layers_to_freeze = self.text_encoder.max_layers_to_freeze
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        if self.vq :
            quantized_graph, quantization_loss_graph = self.quantization(graph_encoded)
            quantized_text, quantization_loss_text = self.quantization(text_encoded)
            return graph_encoded, text_encoded, quantization_loss_graph, quantization_loss_text
        return graph_encoded, text_encoded
    
    def freeze_layers(self, num_layers_to_freeze):
        if self.param['model_name'] == "allenai/scibert_scivocab_uncased":
            self.text_encoder.freeze_layers_scibert(num_layers_to_freeze)
        else:
            self.text_encoder.freeze_layers(num_layers_to_freeze)
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
