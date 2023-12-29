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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class QuantizationLayer(nn.Module):
    def __init__(self, parameters):
        super(QuantizationLayer, self).__init__()
        self.embedding_dim = parameters['nout']
        self.num_embeddings = parameters['num_embeddings']
        self.beta = parameters['beta']
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        init.xavier_uniform_(self.embedding.weight)
        # self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)


# def forward(self, latents: Tensor) -> Tensor:
#         latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
#         latents_shape = latents.shape
#         flat_latents = latents.view(-1, self.D)  # [BHW x D]

#         # Compute L2 distance between latents and embedding weights
#         dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#                torch.sum(self.embedding.weight ** 2, dim=1) - \
#                2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

#         # Get the encoding that has the min distance
#         encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

#         # Convert to one-hot encodings
#         device = latents.device
#         encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
#         encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

#         # Quantize the latents
#         quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
#         quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

#         # Compute the VQ Losses
#         commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
#         embedding_loss = F.mse_loss(quantized_latents, latents.detach())

#         vq_loss = commitment_loss * self.beta + embedding_loss

#         # Add the residue back to the latents
#         quantized_latents = latents + (quantized_latents - latents).detach()

#         return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

    def forward(self, x):
        # Flatten x to (batch_size * nout, embedding_dim)
        # print(x.shape)
        # flat_x = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(x**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(x, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        quantized_detach = (quantized - x).detach() + x
        return quantized_detach, vq_loss

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
        
        return graph_encoded, text_encoded, 1, 1
        normalized_graph_encoded = graph_encoded / graph_encoded.norm(dim=1, keepdim=True)
        normalized_text_encoded = text_encoded / text_encoded.norm(dim=1, keepdim=True)
        
        # embeddings = graph_encoded.detach().cpu().numpy()
        # tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        # tsne_results = tsne.fit_transform(embeddings)

        # plt.figure(figsize=(10, 6))
        # plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        # plt.title('t-SNE visualization of Graph Encoded Embeddings')
        # plt.savefig('graph_encoded_tsne.png')
        # print("fig saved")

        # print(graph_encoded.mean())
        # print(text_encoded.mean())
        
        # Quantization
        quantized_graph, quantization_loss_graph = self.quantization(normalized_graph_encoded)
        quantized_text, quantization_loss_text = self.quantization(normalized_text_encoded)

        return quantized_graph, quantized_text, quantization_loss_graph, quantization_loss_text
