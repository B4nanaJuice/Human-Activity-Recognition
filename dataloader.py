from torch.utils.data import Dataset
import torch
import csv
import numpy as np
import torch.nn.functional as F

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def csv_to_tensor(file_name: str, tensor_size: tuple[int, int] = (64, 64)) -> torch.Tensor:
    """
    Function

    Parameters
    ----------
    file_name : str
        Name of the file to open, usually csv files.
    tensor_size : tuple[int, int]
        Size of the output torch.Tensor. Default value = (64, 64)

    Returns
    --------
    torch.Tensor
        Tensor with the file's content inside (must be a csv file with float values).
    """

    _reader = csv.reader(open(file_name, mode = 'r'), delimiter = ',')
    _x = list(_reader)
    _result = np.array(_x).astype(np.float32)
    _tensor = torch.from_numpy(_result).unsqueeze(0).unsqueeze(0)
    _tensor = F.adaptive_avg_pool2d(_tensor, tensor_size)
    _tensor = _tensor.squeeze()
    return _tensor # Tensor [1, 64, 64]