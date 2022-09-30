import numpy as np


def array_bool_subset(index, min_size=2):
    b = np.zeros(len(index), dtype=bool)
    selected = np.random.choice(
        range(len(index)),
        size=np.random.randint(min_size, len(index), ()),
        replace=False,
    )
    b[selected] = True
    return b


def array_subset(index, min_size=2):
    if len(index) < min_size:
        raise ValueError(f"min_size (={min_size}) must be smaller than len(index) (={len(index)}")
    return np.random.choice(index, size=np.random.randint(min_size, len(index), ()), replace=False)


def array_int_subset(index, min_size=2):
    if len(index) < min_size:
        raise ValueError(f"min_size (={min_size}) must be smaller than len(index) (={len(index)}")
    return np.random.choice(
        np.arange(len(index)),
        size=np.random.randint(min_size, len(index), ()),
        replace=False,
    )


def slice_subset(index, min_size=2):
    while True:
        points = np.random.choice(np.arange(len(index) + 1), size=2, replace=False)
        s = slice(*sorted(points))
        if len(range(*s.indices(len(index)))) >= min_size:
            break
    return s


def single_subset(index):
    return index[np.random.randint(0, len(index))]
