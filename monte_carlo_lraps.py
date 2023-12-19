import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

# Nombre de matrices à générer pour l'estimation
num_matrices = 100

# Taille de la matrice
matrix_size = 3300

def generate_permutation_matrix(size):
    """ Générer une matrice de permutation de taille spécifiée. """
    # Créer une matrice identité
    permutation_matrix = np.eye(size)
    
    # Mélanger les lignes
    np.random.shuffle(permutation_matrix)

    # Mélanger les colonnes
    np.random.shuffle(permutation_matrix.T)

    return permutation_matrix

# Réinitialisation de la liste pour stocker les scores LRAP
lrap_scores_permutation = []

for _ in range(num_matrices):
    # Générer une matrice aléatoire de taille 100x100
    random_matrix_permutation = np.random.rand(matrix_size, matrix_size)

    # Générer une matrice de permutation
    permutation_labels = generate_permutation_matrix(matrix_size)

    # Calculer le LRAP pour cette matrice
    lrap_score_permutation = label_ranking_average_precision_score(permutation_labels, random_matrix_permutation)
    lrap_scores_permutation.append(lrap_score_permutation)

# Calculer la moyenne des scores LRAP pour les matrices de permutation 100x100
average_lrap_score_permutation = np.mean(lrap_scores_permutation)
print(average_lrap_score_permutation)
# return = 0.0026228147431405944