import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data = sio.loadmat('data/dataset_meeting_room.mat')

# _ = data.get('dataset_CSI')
# _ = _[1, :, :]

# plt.plot(_)
# plt.show()


X = data.get('dataset_CSI')
y = data.get('dataset_labels')

print(X.shape)
print(y.shape)

x_means = X.mean(2)
# print(x_means.shape)

# plt.plot(x_means[3, :])
# plt.show()