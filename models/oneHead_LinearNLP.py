from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from transformers import AutoModel, BertModel
from torch.nn import Dropout

from models.quantization import QuantizationLayer

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm, SAGEConv, ChebConv, GINConv, EdgeConv, MessagePassing, TransformerConv

class GraphEncoderOneHead(nn.Module):
    def __init__(self, parameters):            
        super(GraphEncoderOneHead, self).__init__()
        num_node_features = parameters['num_node_features']
        nout = parameters['nout']
        nhid = parameters['nhid']
        graph_hidden_channels = parameters['graph_hidden_channels']
        self.pooling = parameters.get("pool", "mean")
        self.use_dropout = parameters.get("use_dropout", False)
        if self.use_dropout:
            dropout_rate = parameters.get("dropout_rate", 0.5)
            self.dropout = Dropout(p=dropout_rate)
        # mlp_layers = parameters['mlp_layers']
        num_heads = 1
        
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(nout)
        
        # GCN layers with BatchNorm
        if parameters.get("use_sage", False):
            self.conv1 = SAGEConv(num_node_features, graph_hidden_channels)
            self.conv2 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
        elif parameters.get("use_cheb", False):
            self.conv1 = ChebConv(num_node_features, graph_hidden_channels, parameters.get("num_chem", 4))
            self.conv2 = ChebConv(graph_hidden_channels, graph_hidden_channels, parameters.get("num_chem", 4))
            self.conv3 = ChebConv(graph_hidden_channels, graph_hidden_channels, parameters.get("num_chem", 4))
        elif parameters.get("use_gin", False):
            self.conv1 = GINConv(nn.Linear(num_node_features, graph_hidden_channels))
            self.conv2 = GINConv(nn.Linear(graph_hidden_channels, graph_hidden_channels))
            self.conv3 = GINConv(nn.Linear(graph_hidden_channels, graph_hidden_channels))
        elif parameters.get("use_GAT", False):
            self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=num_heads)
            self.conv2 = GATConv(graph_hidden_channels, graph_hidden_channels, heads=num_heads)
            self.conv3 = GATConv(graph_hidden_channels, graph_hidden_channels, heads=num_heads)
        elif parameters.get("use_edge", False):
            self.conv1 = EdgeConv(nn.Linear(num_node_features, graph_hidden_channels))
            self.conv2 = EdgeConv(nn.Linear(graph_hidden_channels, graph_hidden_channels))
            self.conv3 = EdgeConv(nn.Linear(graph_hidden_channels, graph_hidden_channels))
        elif parameters.get("use_mp", False):
            self.conv1 = MessagePassing(nn.Linear(num_node_features, graph_hidden_channels))
            self.conv2 = MessagePassing(nn.Linear(graph_hidden_channels, graph_hidden_channels))
            self.conv3 = MessagePassing(nn.Linear(graph_hidden_channels, graph_hidden_channels))
        elif parameters.get("use_transformers_conv", False):
            self.conv1 = TransformerConv(num_node_features, graph_hidden_channels)
            self.conv2 = TransformerConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = TransformerConv(graph_hidden_channels, graph_hidden_channels)
        else:
            self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
            self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        
        self.bn1 = BatchNorm(graph_hidden_channels)
        self.bn2 = BatchNorm(graph_hidden_channels)
        self.bn3 = BatchNorm(graph_hidden_channels)
        
        # Attention layer with BatchNorm
        self.attention = GATConv(graph_hidden_channels, graph_hidden_channels, heads=num_heads)
        self.bn_attention = BatchNorm(graph_hidden_channels)
        
        # Linear layers
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nhid)
        
        self.mol_hidden4 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        if not self.use_dropout:
            x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
            
            x = self.relu(self.bn1(self.conv1(x, edge_index))) + x
            x = self.relu(self.bn2(self.conv2(x, edge_index))) + x
            x = self.relu(self.bn3(self.conv3(x, edge_index))) + x

            # Apply attention mechanism with residual connection
            x = self.relu(self.bn_attention(self.attention(x, edge_index))) + x

            # Pooling and linear layers
            
            if self.pooling == "add":
                x = global_add_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)
                
            x = self.relu(self.mol_hidden1(x))
            x = self.relu(self.mol_hidden2(x))
            x = self.relu(self.mol_hidden3(x))
            x = self.mol_hidden4(x)

            return x
        else:
            x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
    
            # Convolutional layers with ReLU activation and Dropout
            x = self.relu(self.bn1(self.conv1(x, edge_index))) + x
            x = self.dropout(x)  # Dropout after first convolutional layer
            x = self.relu(self.bn2(self.conv2(x, edge_index))) + x
            x = self.dropout(x)  # Dropout after second convolutional layer
            x = self.relu(self.bn3(self.conv3(x, edge_index))) + x
            x = self.dropout(x)  # Dropout after third convolutional layer

            # Apply attention mechanism with residual connection
            x = self.relu(self.bn_attention(self.attention(x, edge_index))) + x
            x = self.dropout(x)  # Dropout after attention mechanism

            # Pooling
            if self.pooling == "add":
                x = global_add_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)
                
            # Linear layers with ReLU activation and Dropout
            x = self.relu(self.mol_hidden1(x))
            x = self.dropout(x)  # Dropout after first linear layer
            x = self.relu(self.mol_hidden2(x))
            x = self.dropout(x)  # Dropout after second linear layer
            x = self.relu(self.mol_hidden3(x))
            x = self.dropout(x)  # Dropout after third linear layer
            x = self.mol_hidden4(x)  # No dropout before the final output

            return x
    
    def mutate(self, mutation_rate):
        with torch.no_grad():  # Nous ne voulons pas de gradients pendant la mutation
            for param in self.parameters():
                # Crée un masque de la même taille que le paramètre
                # Le masque est vrai avec une probabilité égale au pourcentage
                mask = torch.rand_like(param) < (mutation_rate)

                # Génère de nouveaux poids aléatoires
                new_weights = torch.randn_like(param)

                # Applique la mutation aux poids sélectionnés
                param[mask] = new_weights[mask]
    
class TextEncoder(nn.Module):
    def __init__(self, parameters):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(parameters['model_name'])
        print(self.bert)
        nout = parameters['nout']
        self.linear = nn.Linear(self.bert.config.hidden_size, nout)
        self.norm = nn.LayerNorm(nout)
        num_layers_to_freeze = parameters.get('num_layers_to_freeze', 0)
        if parameters['model_name'] == "allenai/scibert_scivocab_uncased":
            self.max_number_to_freeze_scibert()
        elif parameters['model_name'] == "seyonec/ChemBERTa-zinc-base-v1":
            self.max_number_to_freeze_scibert()
        else:
            self.max_number_to_freeze()
        if num_layers_to_freeze != 0:
            if parameters['model_name'] == "allenai/scibert_scivocab_uncased":
                self.freeze_layers_scibert(num_layers_to_freeze)
            elif parameters['model_name'] == "seyonec/ChemBERTa-zinc-base-v1":
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
        print("The max number of freezable layers is : ",max_layers_to_freeze)
        
    def max_number_to_freeze_scibert(self):
        max_layers_to_freeze = 0
        for layer in self.bert.encoder.layer:
            max_layers_to_freeze += 1
        self.max_layers_to_freeze = max_layers_to_freeze
        print("The max number of freezable layers is : ",max_layers_to_freeze)
        
    def max_number_to_freeze_chem(self):
        max_layers_to_freeze = 0
        for layer in self.encoder.layer:
            max_layers_to_freeze += 1
        self.max_layers_to_freeze = max_layers_to_freeze
        print("The max number of freezable layers is : ",max_layers_to_freeze)

    
    def freeze_layers(self, num_layers_to_freeze):
        frozen_layers = 0
        print("freezing ",num_layers_to_freeze, " layers")
        for layer in self.bert.transformer.layer:
            if frozen_layers < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_layers += 1
            else:
                break
            
    def freeze_layers_chem(self, num_layers_to_freeze):
        frozen_layers = 0
        print("Freezing", num_layers_to_freeze, "layers")
        for layer in self.encoder.layer:
            if frozen_layers < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_layers += 1
            else:
                break
            
    def mutate(self, mutation_rate, layer_to_mutate = 12):
        frozen_layers = 0
        for layer in self.bert.encoder.layer:
            frozen_layers += 1
            if frozen_layers == layer_to_mutate:
                for param in layer.parameters():
                    with torch.no_grad():  # Nous ne voulons pas de gradients pendant la mutation
                        for param in self.parameters():
                            # Crée un masque de la même taille que le paramètre
                            # Le masque est vrai avec une probabilité égale au pourcentage
                            mask = torch.rand_like(param) < (mutation_rate)

                            # Génère de nouveaux poids aléatoires
                            new_weights = torch.randn_like(param)

                            # Applique la mutation aux poids sélectionnés
                            param[mask] = new_weights[mask]
    
    def freeze_layers_scibert(self, num_layers_to_freeze):
        print("freezing ",num_layers_to_freeze, " layers")
        for layer in self.bert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoderOneHead(parameters)
        self.text_encoder = TextEncoder(parameters)
        self.param = parameters
        # print(self.text_encoder)
        self.temp_value = parameters.get("temperature", 0)
        if self.temp_value != 0:
            self.temp = nn.Parameter(torch.Tensor([parameters.get("temperature", 0)])) 
        self.vq = parameters['VQ']
        if self.vq :
            self.quantization = QuantizationLayer(parameters)
        self.max_layers_to_freeze = self.text_encoder.max_layers_to_freeze
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        if self.temp_value !=0:
            graph_encoded = graph_encoded * torch.exp(self.temp)
            text_encoded = text_encoded * torch.exp(self.temp)
        if self.vq :
            quantized_graph, quantization_loss_graph = self.quantization(graph_encoded)
            quantized_text, quantization_loss_text = self.quantization(text_encoded)
            self.quantization_loss = quantization_loss_graph + quantization_loss_text
            return graph_encoded, text_encoded
        return graph_encoded, text_encoded
    
    def freeze_layers(self, num_layers_to_freeze):
        if self.param['model_name'] == "allenai/scibert_scivocab_uncased":
            self.text_encoder.freeze_layers_scibert(num_layers_to_freeze)
        elif self.param['model_name'] == "seyonec/ChemBERTa-zinc-base-v1":
            self.text_encoder.freeze_layers_chem(num_layers_to_freeze)
        else:
            self.text_encoder.freeze_layers(num_layers_to_freeze)
            
    def mutate(self, mutation_rate):
        self.text_encoder.mutate(mutation_rate = mutation_rate)
        # self.graph_encoder.mutate(mutation_rate = mutation_rate)
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
