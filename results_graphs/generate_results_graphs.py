import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Fonction pour extraire les scalaires d'un fichier de log TensorBoard
def extract_tb_scalars(log_path, scalar_tag):
    ea = event_accumulator.EventAccumulator(log_path,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    scalars = ea.Scalars(scalar_tag)
    values = [s.value for s in scalars]
    steps = [s.step for s in scalars]
    return steps, values

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')



# Fonction pour lisser les données (moyenne mobile)
def smooth_data(values, weight=0.3):  # weight détermine le degré de lissage
    last = values[0]
    smoothed = []
    print("---")
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
        print(point, smoothed_val)
    return smoothed

# Spécifiez vos chemins de fichiers de logs
log_files = [
    'logs/logs/scibert_3_freeze_100-2024-01-06--1540',
    'nlogs/logs/scibert_3_freeze_70-2024-01-05--0005',
    'logs/logs/scibert_3_freeze-2024-01-04--1040'
]

# Tag du scalaire à extraire
scalar_tag = 'Lraps/val'


# Ajouter une droite verticale en pointillés à l'abscisse 100


# Ajouter un message
# plt.text(100, max(values), 'Freeze the TextModel after this epoch', rotation=90, verticalalignment='center')
plt.figure(figsize=(10, 6))
for file in log_files:
    steps, values = extract_tb_scalars(file, scalar_tag)
    smoothed_values = moving_average(values)

    plt.plot(steps, smoothed_values, label=os.path.basename(file))  # Courbe lissée
    plt.plot(steps, values, alpha=0.3) 

# Extraire les données et les tracer
plt.figure(figsize=(10, 6))
for file in log_files:
    steps, values = extract_tb_scalars(file, scalar_tag)
    plt.plot(steps, values, label=os.path.basename(file))
    plt.axvline(x=100, color='red', linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('LRAPS')
plt.title('LRAPS vs Epochs')
plt.legend()
plt.savefig("results_graphs/impact_of_the_freeze.png", dpi=300)
