
import logging
import os
import random
import numpy as np
import torch


log = logging.getLogger(__name__)
def seed_everything(seed) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    Args:
        seed: the integer value seed for global random state.
    """
    if not isinstance(seed, int):
        seed = int(seed)

    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed