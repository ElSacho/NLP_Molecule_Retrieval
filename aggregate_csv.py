import csv
import pandas as pd
import csv
import numpy as np
import pandas as pd
import os
from sklearn.metrics import label_ranking_average_precision_score

def variance_reciprocal_csv(file_paths):
    # Initialize lists to store data and IDs
    sum_data = []
    squared_diff = []
    ids = []
    num_files = len(file_paths)

    # First read to initialize sum_data and ids
    with open(file_paths[0], 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ids.append(row[0])  # Save the IDs
            sum_data.append([float(val) for val in row[1:]])  # Initialize sum_data with the first values

    # Read other files and add their data to sum_data
    for file_path in file_paths[1:]:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                sum_data[i] = [x + float(y) for x, y in zip(sum_data[i], row[1:])]

    # Calculate the mean
    mean_data = [[x / num_files for x in row] for row in sum_data]

    # Initialize squared_diff with zeros
    squared_diff = [[0]*len(mean_data[0]) for _ in range(len(mean_data))]

    # Calculate the sum of squared differences from the mean
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                squared_diff[i] = [squared_diff_val + (float(val) - mean_val) ** 2 for squared_diff_val, mean_val, val in zip(squared_diff[i], mean_data[i], row[1:])]

    # Calculate the variance and its reciprocal
    variance_reciprocal_data = []
    for mean_row, var_row in zip(mean_data, squared_diff):
        variance_reciprocal_row = []
        for mean_val, var_val in zip(mean_row, var_row):
            if mean_val > 0.1:
                variance = var_val / num_files
                variance_reciprocal_row.append(1 / variance if variance != 0 else float('inf'))
            else:
                variance_reciprocal_row.append(0)
        variance_reciprocal_data.append(variance_reciprocal_row)

    # Write the result to a new CSV file
    with open('variance_reciprocal.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for id, row in zip(ids, variance_reciprocal_data):
            writer.writerow([id] + row)

def moyenne_csv(file_paths):
    # Initialiser une liste pour stocker toutes les données et les ID
    somme_data = []
    ids = []
    nb_fichiers = len(file_paths)

    # Première lecture pour initialiser la structure de somme_data et ids
    with open(file_paths[0], 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ids.append(row[0])  # Sauvegarder les ID
            somme_data.append([float(val) for val in row[1:]])  # Initialiser somme_data avec les premières valeurs

    # Lire les autres fichiers et ajouter leurs données à somme_data
    for file_path in file_paths[1:]:
        print(file_path)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                somme_data[i] = [x + float(y) for x, y in zip(somme_data[i], row[1:])]

    # Calculer la moyenne
    moyenne_data = [[x / nb_fichiers for x in row] for row in somme_data]

    # Écrire le résultat dans un nouveau fichier CSV
    with open('moyenne.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for id, row in zip(ids, moyenne_data):
            writer.writerow([id] + row)
            
def sum_csv(file_paths):
    # Initialiser une liste pour stocker toutes les données et les ID
    somme_data = []
    ids = []
    nb_fichiers = len(file_paths)

    # Première lecture pour initialiser la structure de somme_data et ids
    with open(file_paths[0], 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ids.append(row[0])  # Sauvegarder les ID
            somme_data.append([float(val) for val in row[1:]])  # Initialiser somme_data avec les premières valeurs

    # Lire les autres fichiers et ajouter leurs données à somme_data
    for file_path in file_paths[1:]:
        print(file_path)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                somme_data[i] = [x + float(y) for x, y in zip(somme_data[i], row[1:])]  

    # Écrire le résultat dans un nouveau fichier CSV
    with open('sum.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for id, row in zip(ids, somme_data):
            writer.writerow([id] + row)

def sum_csv_with_rank_adjustment(initial_csv_path):
    # Initialiser une liste pour stocker toutes les données ajustées et les ID
    somme_data = []
    ids = []
    first_file = True

    with open(initial_csv_path, 'r') as initial_file:
        initial_reader = csv.DictReader(initial_file)
        for row in initial_reader:
            file_path = row['Chemin du fichier']
            rang = int(row['Rang'])  # Assumant que le rang est correctement converti en entier

            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                if first_file:
                    # Initialiser la structure de somme_data et ids avec les premières valeurs ajustées
                    for row in reader:
                        ids.append(row[0])  # Sauvegarder les ID
                        somme_data.append([float(val) * (50 - rang) for val in row[1:]])  # Ajuster par (50 - Rang)
                    first_file = False
                else:
                    # Ajouter les données des autres fichiers à somme_data, ajustées par (50 - Rang)
                    for i, row in enumerate(reader):
                        somme_data[i] = [x + float(y) * (50 - rang) for x, y in zip(somme_data[i], row[1:])]

    # Écrire le résultat dans un nouveau fichier CSV
    with open('sum_adjusted_by_rank.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for id, data in zip(ids, somme_data):
            writer.writerow([id] + data)
            
def total_binary_search(file_paths):
    best_LRAPS = 0
    # best_LRAPS = binary_search_sum(file_paths[0], file_paths[1], 1, best_LRAPS) 
       
    for i in range( 0, len(file_paths) ):
        best_LRAPS = binary_search_sum("binary_best.csv", file_paths[i], i, best_LRAPS)
        
def binary_search_sum(file1, file2, iteration, best_LRAPS, lraps_file1 = None, max_step = 5):
    if lraps_file1 is not None:
        left_LRAPS = lraps_file1
    else:
        left_LRAPS = calculate_LRAPS(file1)
    a = 0
    b = 2 / (iteration + 49)
    t = (a+b)/2
    
    right_LRAPS = weighted_sum(file1, file2, b, best_LRAPS)
    
    print("STARTING THE BINARY SEARCH")
    for i in range(max_step):
        middle_LRAPS = weighted_sum(file1, file2, t, best_LRAPS)
        if max(middle_LRAPS, right_LRAPS, left_LRAPS) == middle_LRAPS:
            if max(right_LRAPS, left_LRAPS) == left_LRAPS:
                b = t
                t = (t + a) / 2
                print('<-', t)
                right_LRAPS = middle_LRAPS
            else:
                a = t
                t = (t + b) / 2
                print('->', t)
                left_LRAPS = middle_LRAPS
        elif max(middle_LRAPS, right_LRAPS, left_LRAPS) == left_LRAPS:
            b = t
            t = (t + a) / 2
            print('<-', t)
            right_LRAPS = middle_LRAPS
        else:
            a = t
            t = (t + b) / 2
            print('->', t)
            left_LRAPS = middle_LRAPS
        best_LRAPS = max(middle_LRAPS, right_LRAPS, left_LRAPS, best_LRAPS)
    print('New file')
    return best_LRAPS

def calculate_LRAPS(file_path):
    chemin_predictions = file_path
    chemin_vrais_labels = 'true_labels.csv' # A np.eye matrix
    
    # Charger les données
    predictions = charger_csv(chemin_predictions)
    # print(predictions[0])
    vrais_labels = charger_csv(chemin_vrais_labels)
    # print(vrais_labels[0])

    for j in range(10):
        # Trouver les indices où vrais_labels[0][i] = 1
        indices = [i for i, label in enumerate(vrais_labels[j]) if label == 1.0]
        # print(indices)

        # Extraire les valeurs correspondantes de predictions[0]
        valeurs_correspondantes = [predictions[j][i] for i in indices]

        # print("Valeurs dans predictions[0] aux indices où vrais_labels[0][i] = 1 :", valeurs_correspondantes)
    # Calculer le score LRAPS
    score = label_ranking_average_precision_score(vrais_labels, predictions)
    print("LRAPS : ", score)
    return score

def charger_csv(chemin_fichier):
    with open(chemin_fichier, newline='') as f:
        lecteur = csv.reader(f)
        next(lecteur, None)  # Ignorer les en-têtes
        return np.array([list(map(float, row[1:])) for row in lecteur])

def adaboost_classifiers(file_paths, true_labels_path, iterations=10):
    n_classifiers = len(file_paths)
    classifier_weights = np.ones(n_classifiers)  # Initialize weights
    vrais_labels = np.array(charger_csv(true_labels_path))

    for iteration in range(iterations):
        # Compute weighted ensemble prediction
        ensemble_prediction = np.zeros_like(vrais_labels, dtype=float)
        for file_index, file_path in enumerate(file_paths):
            classifier_output = np.array(charger_csv(file_path))
            ensemble_prediction += classifier_weights[file_index] * classifier_output

        # Calculate LRAPS for ensemble
        ensemble_score = label_ranking_average_precision_score(vrais_labels, ensemble_prediction)
        print(f"Iteration {iteration}, Ensemble LRAPS: {ensemble_score}")

        # Calculate performance and update weights
        for file_index, file_path in enumerate(file_paths):
            classifier_output = np.array(charger_csv(file_path))
            classifier_score = label_ranking_average_precision_score(vrais_labels, classifier_output)
            error = 1 - classifier_score
            if error < 0.5:
                classifier_weights[file_index] *= (error / (1 - error))

        # Normalize weights
        classifier_weights /= classifier_weights.sum()

def weighted_sum(file1, file2, t, best_LRAPS):
    # Initialiser une liste pour stocker toutes les données ajustées et les ID
    somme_data = []
    ids = []

    with open(file1, 'r') as file:
        reader = csv.reader(file)
        # Initialiser la structure de somme_data et ids avec les premières valeurs ajustées
        for row in reader:
            ids.append(row[0])  # Sauvegarder les ID
            somme_data.append([float(val) * (1-t) for val in row[1:]])  # Ajuster par (50 - Rang)
                
    with open(file2, 'r') as file:
        reader = csv.reader(file)
        # Initialiser la structure de somme_data et ids avec les premières valeurs ajustées
        for i, row in enumerate(reader):
            somme_data[i] = [x + float(y) * t for x, y in zip(somme_data[i], row[1:])]
            
    with open('binary_temp.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for id, data in zip(ids, somme_data):
            writer.writerow([id] + data)
    
    lraps = calculate_LRAPS('binary_temp.csv')
    best_LRAPS = max(best_LRAPS, lraps)
    if best_LRAPS == lraps:
        print("SAVING NEW BEST MODEL")
        with open('binary_best.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for id, data in zip(ids, somme_data):
                writer.writerow([id] + data)
                
    return lraps

folder_paths = ['NLP_Molecule_Retrieval/csv/csv_files']

file_paths = []
for folder_path in folder_paths:
    for file in os.listdir(folder_path):
        if not file.startswith('.') and file!="NLP_Molecule_Retrieval/csv_files/.DS_Store":
            file_path = os.path.join(folder_path, file)
            file_paths.append(file_path)
    

sum_csv(file_paths)
# sum_csv_with_rank_adjustment('rank.csv')
calculate_LRAPS("sum.csv")

