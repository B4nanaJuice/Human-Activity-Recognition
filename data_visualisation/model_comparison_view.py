import matplotlib.pyplot as plt
import json
import numpy as np

## Decide which comparison to make
image_size: str | list[str] = "52x600"
conv_layer: int | list[int] = [2, 3, 4]
fc_layer: int | list[int] = 1

accuracies: list = []
training_time: list = []
losses: list = []

## Choose json files that will be used to compare
files = []

if type(image_size) == list:
    for _ in image_size:
        files.append(f"CNN{_}_{conv_layer}CONV{fc_layer}FC")
elif type(conv_layer) == list:
    for _ in conv_layer:
        files.append(f"CNN{image_size}_{_}CONV{fc_layer}FC")
elif type(fc_layer) == list:
    for _ in fc_layer:
        files.append(f"CNN{image_size}_{conv_layer}CONV{_}FC")
else:
    files.append(f"CNN{image_size}_{conv_layer}CONV{fc_layer}FC")

## Import data from each json file
for file in files:
    # Import file
    with open(f"results/{file}.json", 'r') as f:
        data = json.loads(f.read())

    # Get data and put it into lists
    accuracies.append(data["test"]["accuracy"])
    training_time.append(data["train"]["training_time"])
    losses.append(np.array(data["train"]["losses"]).mean(0))

## Make plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
runs = [_+1 for _ in range(data["hyperparameters"]["num_runs"])]
epochs = [_+1 for _ in range(data["hyperparameters"]["epochs"])]

acc_legend = []
tt_legend = []
loss_legend = []

for _ in range(len(files)):
    # Accuracy
    average_acc = round(sum(accuracies[_]) / len(runs), 2)
    ax1.plot(runs, accuracies[_], '-', runs, [average_acc] * len(runs), '--')

    acc_legend.append(files[_])
    acc_legend.append(f"Average - {average_acc}%")

    # Training time
    average_tt = round(sum(training_time[_]) / len(runs), 2)
    ax2.plot(runs, training_time[_], '-', runs, [average_tt] * len(runs), '--')

    tt_legend.append(files[_])
    tt_legend.append(f"Average - {average_tt}s")

    # Loss
    ax3.semilogy(epochs, losses[_])

    loss_legend.append(files[_])

ax1.set_xlabel('Run index')
ax1.set_ylabel('Accuracy (%)')

ax2.set_xlabel('Run index')
ax2.set_ylabel('Time taken (s)')
ax2.set_title(f'Accuracy, training time and training loss comparison for models {", ".join(files)}')

ax3.set_xlabel('Epoch index')
ax3.set_ylabel('Training loss')

ax1.legend(acc_legend)
ax2.legend(tt_legend)
ax3.legend(loss_legend)

## Show the plots
plt.show()