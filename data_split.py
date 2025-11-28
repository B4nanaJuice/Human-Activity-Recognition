import numpy as np

def split_data(X, y):
    _data = np.concatenate((X, y), axis = 1)
    np.random.shuffle(_data)
    _num_rows_percentage = 10
    _num_rows = int(_num_rows_percentage * _data.shape[0] / 100)

    X_train = _data[_num_rows:, :-1]
    y_train = _data[_num_rows:, -1:]
    X_test = _data[:_num_rows, :-1]
    y_test = _data[:_num_rows, -1:]

    return X_train, y_train, X_test, y_test