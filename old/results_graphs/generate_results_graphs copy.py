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
    return smoothed

# Spécifiez vos chemins de fichiers de logs
# log_files = [
#     'logs/logs/scibert_3_freeze_100-2024-01-06--1540',
#     'nlogs/logs/scibert_3_freeze_70-2024-01-05--0005',
#     'logs/logs/scibert_3_freeze-2024-01-04--1040'
# ]

log_files = get_all_folder_paths("final_logs/logs/conv")
print(log_files)

# Tag du scalaire à extraire
scalar_tag = 'Lraps/val'

plt.figure(figsize=(10, 6))

# Plotting the main graph
for file in log_files:
    try:
        steps, values = extract_tb_scalars(file, scalar_tag)
        # values = values[:110]
        # steps = steps[:110]
        smoothed_values = smooth_data(values)

        # plt.plot(steps, smoothed_values, label=os.path.basename(file))  # Smoothed curve
        label = os.path.basename(file)
        print(label)
        label = label.split('-')[0]
        plt.plot(steps, values, label=label)  # Original data
    except:
        pass

plt.xlabel('Epochs')
plt.ylabel('LRAPS on the validation')
plt.title('Impact of the convolutional layer')

# Add the legend in the lower right. Adjust `bbox_to_anchor` if necessary
# legend = plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
legend = plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=2)

# plt.legend(loc='lower right')

# Add an inset (zoomed area) just above the legend in the lower right
# You'll need to manually adjust `bbox_to_anchor` for the legend and `loc` for the inset
axins = inset_axes(plt.gca(), width="40%", height="40%", loc='lower right', bbox_to_anchor=(0, 0.3, 1, 1), bbox_transform=plt.gca().transAxes)

for file in log_files:
    try:
        steps, values = extract_tb_scalars(file, scalar_tag)
        # values = values[:110]
        # steps = steps[:110]
        smoothed_values = smooth_data(values, weight=0.9)

        if len(steps) > 50:
            zoom_range = slice(-50, None)  # Last 50 elements
        else:
            zoom_range = slice(0, None)
            
        # label = os.path.basename(file)
        # print(label)
        # label = label.split('-')[0]
        # plt.plot(steps, values, label=label)  # Original data
        print(file)
        axins.plot(steps[zoom_range], smoothed_values[zoom_range])
        axins.plot(steps[zoom_range], values[zoom_range], alpha=0.2, label= label)
    except:
        pass
plt.savefig("figures/conv.png", dpi=300)

#temp
# #loss
# conv