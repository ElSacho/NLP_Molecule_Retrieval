import torch
import torch.nn.functional as F


###########################
####### TRIPLET LOSS ######
###########################

def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2.unsqueeze(1), dim=2)
  
def triplet_loss_cosine(anchor, positive, negative, margin):
    """
    Compute the triplet loss using cosine similarity.

    Parameters:
    anchor (Tensor): The anchor embeddings.
    positive (Tensor): The positive embeddings.
    negative (Tensor): The negative embeddings.
    margin (float): The margin value for the loss calculation.

    Returns:
    Tensor: The computed triplet loss.
    """
    # Calcul de la similarité cosinus
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    pos_sim = cos_sim(anchor, positive)
    neg_sim = cos_sim(anchor, negative)

    # Calcul de la triplet loss
    loss = torch.mean(torch.clamp(neg_sim - pos_sim + margin, min=0.0))

    return loss

def triplet_loss_sim(x_graph, x_text, margin):
    hard_negative_triplets = create_hard_negative_triplets(x_graph, x_text)
    # Calcul de la perte pour chaque triplet
    # print('triplet : ',hard_negative_triplets)
    hard_negative_losses = [triplet_loss_cosine(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0), margin=margin) 
                            for anchor, positive, negative in hard_negative_triplets]

    # Calcul de la perte moyenne pour les triplets avec négatifs difficiles
    mean_hard_negative_loss = torch.mean(torch.stack(hard_negative_losses))
    return mean_hard_negative_loss
    
def create_hard_negative_triplets(x_graph, x_text):
    """
    Create triplets (anchor, positive, negative) with hard negatives based on cosine similarity.

    Parameters:
    x_graph (Tensor): The graph embeddings.
    x_text (Tensor): The text embeddings.

    Returns:
    List[Tuple[Tensor, Tensor, Tensor]]: The list of triplets.
    """
    n = x_graph.size(0)  # Taille du batch
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    triplets = []

    for i in range(n):
        anchor = x_graph[i]
        positive = x_text[i]

        # Calculer la similarité cosinus entre l'ancrage et tous les exemples de x_text
        similarities = cos_sim(anchor.unsqueeze(0), x_text)

        # Ignorer la similarité avec l'exemple positif correspondant
        similarities[i] = -1

        # Trouver l'indice de l'exemple négatif le plus difficile
        negative_index = torch.argmax(similarities).item()
        negative = x_text[negative_index]

        triplets.append((anchor, positive, negative))

    return triplets
    
def cosine_similarity_loss(v1, v2):
    # Compute the cosine similarity for each pair of vectors
    # v1 has shape [batch_size, features], v2 has shape [batch_size, features]
    # We unsqueeze v1 to [batch_size, 1, features] and v2 to [batch_size, features, 1]
    # to get a [batch_size, batch_size] similarity matrix
    similarity_matrix = F.cosine_similarity(v1.unsqueeze(1), v2.unsqueeze(2), dim=2)

    # Create labels for the diagonal elements, which should be the highest
    labels = torch.arange(similarity_matrix.shape[0], device=v1.device)

    # Use Cross Entropy Loss - this expects logits, not probabilities, so we don't apply softmax
    # The diagonal elements (same pairs) should be maximized, others minimized
    loss = F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.t(), labels)

    return loss

###########################
######  CONTRASTIVE  ######
###########################

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

BCEL = torch.nn.BCEWithLogitsLoss()

def negative_sampling_contrastive_loss(v1, v2, labels):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye), logits.diag() > 0
 
###########################
####### LIFTED LOSS #######
###########################

def lifted_structured_loss(x_graph, x_text, margin):
    """
    Compute the Lifted Structured Loss for embeddings.

    Parameters:
    x_graph (Tensor): The graph embeddings.
    x_text (Tensor): The text embeddings.
    margin (float): The margin value for the loss calculation.

    Returns:
    Tensor: The computed Lifted Structured Loss.
    """
    n = x_graph.size(0)  # Batch size
    labels = torch.arange(n)  # Assuming each pair (x_graph, x_text) is positive

    # Compute pairwise distance matrix
    distance_matrix = compute_pairwise_cosine_distances(x_graph, x_text)

    # Compute loss
    loss = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                # Positive pair contribution
                d_pos = distance_matrix[i, j]
                # Subtracting margin from all negative distances
                neg_distances = distance_matrix[i][labels != labels[j]] - margin
                # Adding positive distance back to ensure it's not considered in the negative part
                neg_distances = torch.cat([neg_distances, d_pos.unsqueeze(0)])
                # Clamping to ensure positive part contributes only if it's within margin
                pos_contribution = F.relu(d_pos + margin - torch.min(neg_distances))
                loss += pos_contribution ** 2
            else:
                # Negative pair contribution is handled within positive pair computation
                continue

    # Finalize loss
    loss = loss / (2 * n)  # Normalizing by the number of pairs
    return loss

def compute_pairwise_distances(x_graph, x_text):
    """
    Compute the pairwise distances between two sets of embeddings.

    Parameters:
    x_graph (Tensor): The graph embeddings.
    x_text (Tensor): The text embeddings.

    Returns:
    Tensor: Pairwise distances.
    """
    # Expanding x_graph and x_text to form a matrix of pairwise distances
    x_graph_square = torch.sum(x_graph ** 2, dim=1).reshape(-1, 1)
    x_text_square = torch.sum(x_text ** 2, dim=1).reshape(1, -1)
    cross_term = 2.0 * torch.matmul(x_graph, x_text.t())

    # Calculating pairwise Euclidean distance
    distances = x_graph_square + x_text_square - cross_term
    distances = torch.sqrt(F.relu(distances))  # Ensuring non-negative distances
    return distances

def compute_pairwise_cosine_distances(x_graph, x_text):
    """
    Compute the pairwise cosine distances between two sets of embeddings, using cosine similarity.

    Parameters:
    x_graph (Tensor): The graph embeddings.
    x_text (Tensor): The text embeddings.

    Returns:
    Tensor: Pairwise cosine distances.
    """
    # Normalize embeddings to unit vectors
    x_graph_norm = F.normalize(x_graph, p=2, dim=1)
    x_text_norm = F.normalize(x_text, p=2, dim=1)

    # Compute cosine similarity matrix
    cosine_sim_matrix = torch.matmul(x_graph_norm, x_text_norm.t())

    # Convert cosine similarities to distances
    cosine_dist_matrix = 1 - cosine_sim_matrix

    return cosine_dist_matrix

 
###########################
######## OLD LOSSES #######
###########################
  
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

def adversarial_loss2(discriminator, x_graph, x_text): 
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

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand_as(real_samples).to(real_samples.device)

    # Interpolate between real and fake data
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    # Calculate probability for interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(real_samples.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Gradients have shape (batch_size, num_features), so flatten to compute norm
    gradients = gradients.view(gradients.size(0), -1)

    # Calculate the L2 norm of the gradients
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradient_norm - 1) ** 2).mean()

def wgan_gp_loss(D, real_samples, fake_samples, lambda_gp):  ## For the GAN network
    """Calculates the WGAN-GP loss"""
    # Calculate probabilities on real and fake data
    real_prob = D(real_samples)
    fake_prob = D(fake_samples)

    # Calculate the WGAN loss
    wgan_loss = fake_prob.mean() - real_prob.mean()

    # Calculate gradient penalty
    gradient_penalty = compute_gradient_penalty(D, real_samples, fake_samples)

    # Return the total loss
    return wgan_loss + lambda_gp * gradient_penalty

