import matplotlib.pyplot as plt
import json

## Overall data
image_size = "64x64"
conv_layers = 4
fc_layers = 1

file_name = f"CNN{image_size}_{conv_layers}CONV{fc_layers}FC.json"
with open(f"results/80-20/{file_name}", 'r') as f:
    data = json.loads(f.read())

num_runs = data["hyperparameters"]["num_runs"]
runs = [_+1 for _ in range(num_runs)]

num_epochs = data["hyperparameters"]["epochs"]
epochs = [_+1 for _ in range(num_epochs)]

classes = ['bend', 'fall', 'lie down', 'run', 'sitdown', 'standup', 'walk']

## Create figures
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Training time
training_time = data["train"]["training_time"]
average_tt = round(sum(training_time) / num_runs, 2)

ax1.plot(runs, training_time, '-', runs, [average_tt] * num_runs, '--')
ax1.set_xlabel('Run index')
ax1.set_ylabel('Time taken (s)')
ax1.set_xticks([_+1 for _ in range(num_runs)])
ax1.set_title('Training time')
ax1.legend(['Measurments', f'Average ({average_tt}s)'])
ax1.grid(visible = True)

# Training loss
losses = data["train"]["losses"]
for _ in range(num_runs):
    ax2.semilogy(epochs, losses[_])
ax2.set_xlabel('Epoch index')
ax2.set_ylabel('Loss')
ax2.set_title('Training loss over epochs')
ax2.grid(visible = True)

# Evaluation accuracy
accuracy = data["test"]["accuracy"]
average_acc = round(sum(accuracy) / num_runs, 2)

ax3.plot(runs, accuracy, '-', runs, [average_acc] * num_runs, '--')
ax3.set_xlabel('Run index')
ax3.set_ylabel('Accuracy (%)')
ax3.set_xticks([_+1 for _ in range(num_runs)])
ax3.set_title('Model evaluation')
ax3.set_ylim([50, 100])
ax3.legend(['Measurments', f'Average ({average_acc}%)'])
ax3.grid(visible = True)

# Confusion matrix
conf_matrix = data["test"]["confusion_matrix"]

pos = ax4.matshow(conf_matrix)
ax4.set_xlabel('Predicted label')
ax4.set_ylabel('True label')
ax4.set_xticks([_ for _ in range(7)], classes, rotation = 90)
ax4.set_yticks([_ for _ in range(7)], classes)
ax4.tick_params(axis = 'x', bottom = True, top = False, labelbottom = True, labeltop = False)
plt.setp([tick.label1 for tick in ax4.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="center", rotation_mode="anchor")
ax4.set_title(f'Confusion matrix on {num_runs} runs')
fig.colorbar(pos, ax = ax4)

for i in range(len(classes)):
    for j in range(len(classes)):
        c = conf_matrix[j][i]
        ax4.text(i, j, str(c), va = 'center', ha = 'center', color = 'white')

# Show plot 
plt.title = data["hyperparameters"]["model"]
plt.show()