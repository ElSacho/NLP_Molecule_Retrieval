import csv

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
            
# Exemple d'utilisation
file_paths = ["submisisons/distil_both.csv", 'submisisons/scibert_both_load.csv']
moyenne_csv(file_paths)
