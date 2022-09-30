from typing import Tuple, Union

import numpy as np
from scipy.sparse import spmatrix


Index1D = Union[slice, int, np.int64, np.ndarray]
Index = Union[Index1D, Tuple[Index1D, Index1D], spmatrix]
