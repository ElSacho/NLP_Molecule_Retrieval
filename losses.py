import torch
import torch.nn.functional as F

def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2.unsqueeze(1), dim=2)

def compute_triplet_loss(x_graph, x_text, discriminator, margin):
    """
    Computes the bi-directional triplet loss using the hardest negative samples, 
    determined by a discriminator.

    :param x_graph: Batched embeddings for graph (molecules).
    :param x_text: Batched embeddings for text.
    :param discriminator: Discriminator model to score embeddings.
    :param margin: Margin parameter delta for the triplet loss.
    :return: Triplet loss value.
    """
    batch_size = x_graph.size(0)

    # Score embeddings using the discriminator
    scores_graph = discriminator(x_graph)
    scores_text = discriminator(x_text)

    # Compute similarity matrices
    sim_matrix_graph = cosine_similarity(x_graph, x_graph)  # Similarity matrix for graph embeddings
    sim_matrix_text = cosine_similarity(x_text, x_text)    # Similarity matrix for text embeddings

    # Mask for positive samples
    mask = torch.eye(batch_size, dtype=torch.bool, device=x_graph.device)

    # Compute hardest negative for each sample in the batch
    # For graph as anchor
    negative_mask_graph = ~mask
    neg_scores_graph = scores_graph[negative_mask_graph].view(batch_size, -1)
    hardest_negatives_graph = neg_scores_graph.max(dim=1)[0]

    # For text as anchor
    negative_mask_text = ~mask
    neg_scores_text = scores_text[negative_mask_text].view(batch_size, -1)
    hardest_negatives_text = neg_scores_text.max(dim=1)[0]

    # Compute triplet loss for graph and text as anchors
    positive_sim_graph = sim_matrix_graph[mask]
    graph_loss = torch.clamp(margin + hardest_negatives_graph - positive_sim_graph, min=0).mean()

    positive_sim_text = sim_matrix_text[mask]
    text_loss = torch.clamp(margin + hardest_negatives_text - positive_sim_text, min=0).mean()

    # Combine the losses
    total_loss = (graph_loss + text_loss) / 2

    return total_loss

def compute_triplet_loss2(x_graph, x_text, margin):
    """
    Computes the bi-directional triplet loss for batched graph and text embeddings.

    :param x_graph: Batched embeddings for graph (molecules).
    :param x_text: Batched embeddings for text.
    :param margin: Margin parameter delta for the triplet loss.
    :return: Triplet loss value.
    """
    batch_size = x_graph.size(0)

    # Compute similarity matrices
    sim_matrix_graph = cosine_similarity(x_graph, x_graph)  # Similarity matrix for graph embeddings
    sim_matrix_text = cosine_similarity(x_text, x_text)    # Similarity matrix for text embeddings

    # Create masks for positive and negative samples
    mask = torch.eye(batch_size, dtype=torch.bool, device=x_graph.device)
    positive_mask_graph = mask
    negative_mask_graph = ~mask
    positive_mask_text = mask
    negative_mask_text = ~mask

    # Compute positive and negative similarities for graph and text
    positive_sim_graph = sim_matrix_graph[positive_mask_graph].view(batch_size, -1)
    negative_sim_graph = sim_matrix_graph[negative_mask_graph].view(batch_size, -1)
    positive_sim_text = sim_matrix_text[positive_mask_text].view(batch_size, -1)
    negative_sim_text = sim_matrix_text[negative_mask_text].view(batch_size, -1)

    # Compute triplet loss for graph as anchor
    graph_loss = torch.clamp(margin + negative_sim_graph - positive_sim_graph.unsqueeze(1), min=0).mean()

    # Compute triplet loss for text as anchor
    text_loss = torch.clamp(margin + negative_sim_text - positive_sim_text.unsqueeze(1), min=0).mean()

    # Combine the losses
    total_loss = (graph_loss + text_loss) / 2

    return total_loss


# def cosine_similarity(x1, x2):
#     return F.cosine_similarity(x1, x2, dim=-1)

# def compute_triplet_loss(anchor, positive, negative, margin):
#     """
#     Computes the bi-directional triplet loss.

#     :param anchor: Embeddings for the anchor samples.
#     :param positive: Embeddings for the positive samples.
#     :param negative: Embeddings for the negative samples.
#     :param margin: Margin parameter delta for the triplet loss.
#     :return: Triplet loss value.
#     """
#     # Calculate cosine similarities
#     pos_similarity = cosine_similarity(anchor, positive)
#     neg_similarity = cosine_similarity(anchor, negative)

#     # Triplet loss calculation
#     triplet_loss = torch.clamp(neg_similarity - pos_similarity + margin, min=0)

#     # Bi-directional loss (consider both directions)
#     return triplet_loss.mean()

def adversarial_loss(discriminator, x_graph, x_text):
    """
    Compute the adversarial loss for the discriminator and the generator.

    Parameters:
    - discriminator: The discriminator network.
    - x_graph: The molecule (graph) representations.
    - x_text: The text representations.

    Returns:
    - loss: The computed adversarial loss.
    """

    # Labels for real (graph) and fake (text) samples
    real_labels = torch.ones(x_graph.size(0), 1).to(x_graph.device)
    fake_labels = torch.zeros(x_text.size(0), 1).to(x_text.device)

    # Discriminator loss for real (graph) samples
    real_preds = discriminator(x_graph)
    real_loss = F.binary_cross_entropy_with_logits(real_preds, real_labels)

    # Discriminator loss for fake (text) samples
    fake_preds = discriminator(x_text)
    fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_labels)

    # Total loss is the sum of real and fake losses
    total_loss = real_loss + fake_loss

    return total_loss