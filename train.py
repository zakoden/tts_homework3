import collections
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    


if __name__ == "__main__":
    main()
