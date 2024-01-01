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