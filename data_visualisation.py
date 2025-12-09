import matplotlib.pyplot as plt
import json
from sklearn.metrics import ConfusionMatrixDisplay

## Det data from file
image_size = "64x64"
conv_layers = 2
fc_layers = 2

file_name = f"CNN{image_size}_{conv_layers}CONV{fc_layers}FC.json"
with open(f"results/{file_name}", 'r') as f:
    data = json.loads(f.read())

num_runs = data["hyperparameters"]["num_runs"]


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Taille totale de la figure
fig = plt.figure(figsize=(12, 6))

# Grille 2 x 3 (6 cases)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.3)

ax1 = fig.add_subplot(gs[:, 0])
ax1.matshow(data["test"]["confusion_matrix"])

training_time_mean = round(sum(data["train"]["training_time"]) / num_runs, 2)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot([_+1 for _ in range(num_runs)], data["train"]["training_time"], "-", [_+1 for _ in range(num_runs)], [training_time_mean]*num_runs, "--")
ax2.set_xlabel('Run')
ax2.set_ylabel('Training time (s)')
ax2.set_title('Training part')
ax2.legend(['Measurments', f'Mean ({training_time_mean}s)'])

accuracy_mean = round(sum(data["test"]["accuracy"]) / num_runs, 2)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot([_+1 for _ in range(num_runs)], data["test"]["accuracy"], '-', [_+1 for _ in range(num_runs)], [accuracy_mean]*num_runs, '--')
ax3.set_xlabel('Run')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Evaluation part')
ax3.set_ylim([50, 100])
ax3.legend(['Measurments', f'Mean ({accuracy_mean}%)'])

# Un long plot en bas à droite
ax4 = fig.add_subplot(gs[1, 1:3])

plt.show()