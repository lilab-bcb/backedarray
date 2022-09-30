from ast import Index


class DenseArray:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx: Index):
        return self.dataset[idx]

    def __setitem__(self, idx: Index, value):
        self.dataset[idx] = value
