import random

import numpy as np
import paddle
from paddle_geometric import seed_everything


def test_seed_everything():
    seed_everything(0)
    assert random.randint(0, 100) == 49
    assert random.randint(0, 100) == 97
    assert np.random.randint(0, 100) == 44
    assert np.random.randint(0, 100) == 47
    assert int(paddle.randint(low=0, high=100, shape=(1,))) == 41
    assert int(paddle.randint(low=0, high=100, shape=(1,))) == 64
