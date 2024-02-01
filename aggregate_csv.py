import csv

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


# sum_csv_with_rank_adjustment('rank.csv')

# Exemple d'utilisation
import os

folder_paths = [ 'NLP_Molecule_Retrieval/submissions', 'NLP_Molecule_Retrieval/csv_files' ]
# folder_paths = ['NLP_Molecule_Retrieval/submissions']
folder_paths = ['NLP_Molecule_Retrieval/more_0_88']

file_paths = []
for folder_path in folder_paths:
    for file in os.listdir(folder_path):
        if not file.startswith('.') and file!="NLP_Molecule_Retrieval/csv_files/.DS_Store":
            file_path = os.path.join(folder_path, file)
            file_paths.append(file_path)
# file_paths = ["moyenne_distil_sci.csv", "moyenne.csv"]
print(file_paths)
file_paths.append("sum02.csv")
file_paths.append("sum03.csv")
file_paths = ["sum02.csv", "sum03.csv"]
file_paths = ["sum_adjusted_by_rank.csv", "sum_adjusted_by_rank1.csv"]
# file_paths.append("sum01.csv")
sum_csv(file_paths)

# file_paths = ["NLP_Molecule_Retrieval/submissions/cheb2.csv","NLP_Molecule_Retrieval/submissions/cheb3.csv","NLP_Molecule_Retrieval/submissions/cheb6.csv","NLP_Molecule_Retrieval/submissions/cheb8.csv","NLP_Molecule_Retrieval/submissions5/temp_0_07.csv","NLP_Molecule_Retrieval/submissions5/sage.csv","NLP_Molecule_Retrieval/submissions5/LR_8e05.csv","NLP_Molecule_Retrieval/submissions5/decay20.csv","NLP_Molecule_Retrieval/submissions5/cheb4.csv","NLP_Molecule_Retrieval/csv_files/distil_both0_87.csv", "NLP_Molecule_Retrieval/submitted/distil_both_hyper_LR_0_8834.csv"]
