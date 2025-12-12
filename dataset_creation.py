from dataloader import csv_to_tensor
import torch
import os
import numpy as np

# Import data and create data loader
tensor_size = (64, 64)
data_path = 'data'
_result: list[tuple[str, torch.Tensor]] = []
for _sub_directory in os.scandir(data_path):

    # Make sure the path points to a folder and not a file
    if not _sub_directory.is_dir():
        pass

    _label: str = _sub_directory.name
    for _file in os.scandir(f'{data_path}\\{_sub_directory.name}'):
        _result.append((_label, csv_to_tensor(f'{data_path}\\{_sub_directory.name}\\{_file.name}', tensor_size = tensor_size)))

label_to_id = {
    'bend': 0,
    'fall': 1,
    'lie down': 2,
    'run': 3,
    'sitdown': 4,
    'standup': 5,
    'walk': 6,
}

def create_split_dataset():
    # Compute the number of rows that will be used to train the model
    test_percentage = .2
    _test_rows = int(len(_result) * test_percentage)

    # Shuffle the data and take the rows
    np.random.shuffle(_result)
    _train, _test = _result[_test_rows:], _result[:_test_rows]

    _train_tensor, _train_labels = [_[1] for _ in _train], torch.from_numpy(np.array([label_to_id[_[0]] for _ in _train]))
    _test_tensor, _test_labels = [_[1] for _ in _test], torch.from_numpy(np.array([label_to_id[_[0]] for _ in _test]))

    torch.save(_train_tensor, f'tensors/train_tensors_{tensor_size[0]}x{tensor_size[1]}.pt')
    torch.save(_train_labels, f'tensors/train_labels_{tensor_size[0]}x{tensor_size[1]}.pt')
    torch.save(_test_tensor, f'tensors/test_tensors_{tensor_size[0]}x{tensor_size[1]}.pt')
    torch.save(_test_labels, f'tensors/test_labels_{tensor_size[0]}x{tensor_size[1]}.pt')


def create_full_dataset():
    labels = torch.from_numpy(np.array([label_to_id[_[0]] for _ in _result]))
    tensors = [_[1] for _ in _result]

    torch.save(tensors, f'tensors/tensors_{tensor_size[0]}x{tensor_size[1]}.pt')
    torch.save(labels, f'tensors/labels.pt')