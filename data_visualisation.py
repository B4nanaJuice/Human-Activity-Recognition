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


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Taille totale de la figure
fig = plt.figure(figsize=(12, 6))

# Grille 2 x 3 (6 cases)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.3)

# Grand plot à gauche : occupe les 2 lignes de la 1ère colonne
ax1 = fig.add_subplot(gs[:, 0])  

# Deux petits plots en haut à droite
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Un long plot en bas à droite
ax4 = fig.add_subplot(gs[1, 1:3])

# Juste pour voir les cadres
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)

plt.show()