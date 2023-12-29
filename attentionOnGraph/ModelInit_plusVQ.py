import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel
import torch.nn.init as init


class QuantizationLayer(nn.Module):
    def __init__(self, parameters):
        super(QuantizationLayer, self).__init__()
        self.embedding_dim = parameters['embedding_dim']
        self.num_embeddings = parameters['num_embeddings']
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        init.xavier_uniform_(self.embedding.weight)
        # self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        # Flatten x to (batch_size * height * width, embedding_dim)
        flat_x = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # encoding_indices_list = encoding_indices.tolist()
        # print(encoding_indices_list)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and reshape
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(x)
        return quantized, (quantized - x).detach() + x

class GraphEncoder(nn.Module):
    def __init__(self, parameters):
        super(GraphEncoder, self).__init__()
        num_node_features = parameters['num_node_features']
        nout = parameters['nout']
        nhid = parameters['nhid']
        graph_hidden_channels = parameters['graph_hidden_channels']
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, parameters):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(parameters['model_name'])
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(parameters)
        self.text_encoder = TextEncoder(parameters)
        self.quantization = QuantizationLayer(parameters)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        normalized_graph_encoded = graph_encoded / graph_encoded.norm(dim=1, keepdim=True)
        normalized_text_encoded = text_encoded / text_encoded.norm(dim=1, keepdim=True)


        # print(graph_encoded.mean())
        # print(text_encoded.mean())
        
        # Quantization
        quantized_graph, quantization_loss_graph = self.quantization(normalized_graph_encoded)
        quantized_text, quantization_loss_text = self.quantization(normalized_text_encoded)

        return quantized_graph, quantized_text, quantization_loss_graph, quantization_loss_text
