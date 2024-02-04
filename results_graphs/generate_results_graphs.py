import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def get_all_folder_paths(folder):
    all_paths = []
    for entry in os.listdir(folder):
        full_path = os.path.join(folder, entry)
        if os.path.isdir(full_path):
            all_paths.append(full_path)
    return all_paths

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
# log_files = [
#     'logs/logs/scibert_3_freeze_100-2024-01-06--1540',
#     'nlogs/logs/scibert_3_freeze_70-2024-01-05--0005',
#     'logs/logs/scibert_3_freeze-2024-01-04--1040'
# ]

log_files = get_all_folder_paths("final_logs/logs/scibert_LR")
print(log_files)

# Tag du scalaire à extraire
scalar_tag = 'Lraps/val'


# Ajouter une droite verticale en pointillés à l'abscisse 100


# Ajouter un message
# plt.text(100, max(values), 'Freeze the TextModel after this epoch', rotation=90, verticalalignment='center')
# plt.figure(figsize=(10, 6))
# for file in log_files:
#     steps, values = extract_tb_scalars(file, scalar_tag)
#     smoothed_values = smooth_data(values)

#     plt.plot(steps, smoothed_values, label=os.path.basename(file))  # Courbe lissée
#     plt.plot(steps, values, alpha=0.3) 

# plt.xlabel('Epochs')
# plt.ylabel('LRAPS')
# plt.title('LRAPS vs Epochs')
# plt.legend()
# plt.savefig("figures/impact_of_the_freeze.png", dpi=300)

plt.figure(figsize=(10, 6))

# Plotting the main graph
for file in log_files:
    steps, values = extract_tb_scalars(file, scalar_tag)
    smoothed_values = smooth_data(values)

    plt.plot(steps, smoothed_values, label=os.path.basename(file))  # Smoothed curve
    plt.plot(steps, values, alpha=0.3)  # Original data

plt.xlabel('Epochs')
plt.ylabel('LRAPS')
plt.title('LRAPS vs Epochs')

# Add the legend in the lower right. Adjust `bbox_to_anchor` if necessary
legend = plt.legend(loc='lower right', bbox_to_anchor=(1, 0))

# Add an inset (zoomed area) just above the legend in the lower right
# You'll need to manually adjust `bbox_to_anchor` for the legend and `loc` for the inset
axins = inset_axes(plt.gca(), width="20%", height="20%", loc='lower right', bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=plt.gca().transAxes)

for file in log_files:
    steps, values = extract_tb_scalars(file, scalar_tag)
    smoothed_values = smooth_data(values)

    if len(steps) > 50:
        zoom_range = slice(-50, None)  # Last 50 elements
    else:
        zoom_range = slice(0, None)

    axins.plot(steps[zoom_range], smoothed_values[zoom_range], label=os.path.basename(file))
    axins.plot(steps[zoom_range], values[zoom_range], alpha=0.3)

# You may need to adjust these parameters to get the exact layout you want
# axins.set_xlim(...)  # Set if you want to focus on a specific range
# axins.set_ylim(...)  # Set y-axis limits to focus on a specific value range
# axins.set_xlabel('Epochs', fontsize=8)  # Set label with smaller font if needed
# axins.set_ylabel('LRAPS', fontsize=8)  # Set label with smaller font if needed
# axins.tick_params(axis='both', which='major', labelsize=8)  # Smaller tick labels

plt.savefig("figures/impact_of_the_freeze.png", dpi=300)

# This ensures the inset does not block viewing the entire curve but also appears over the legend's original position
# Adjust as necessary for clarity and visibility

# Optional: Adjust the inset's appearance
# axins.set_xlim(...)  # Set if you want to focus on a specific range
# axins.set_ylim(...)  # Set y-axis limits to focus on a specific value range
# axins.set_xlabel('Epochs', fontsize=8)  # Set label with smaller font if needed
# axins.set_ylabel('LRAPS', fontsize=8)  # Set label with smaller font if needed
# axins.tick_params(axis='both', which='major', labelsize=8)  # Smaller tick labels


# Extraire les données et les tracer
# plt.figure(figsize=(10, 6))
# for file in log_files:
#     steps, values = extract_tb_scalars(file, scalar_tag)
#     plt.plot(steps, values, label=os.path.basename(file))
#     plt.axvline(x=100, color='red', linestyle='--')