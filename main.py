import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('data/dataset_meeting_room.mat')

_ = data.get('dataset_CSI')
_ = _[1, :, :]

plt.plot(_)
plt.show()
