import os
import json
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random
import torch


def saveDict(config, path='.', filename='config'):
    """
    A simple code to save a dictionary as a json file
    """
    json.dump(config,
              open(f'{path}/{filename}.json', 'w'))
    print(f'Config saved as: {path}/{filename}.json')


def loadDict(path='.', filename='config'):
    """
    Load a saved dictionary (JSON file)
    """
    if not os.path.exists(f'{path}/{filename}.json'):
        print('Config file not found.')
        return None
    
    config = json.load(open(f'{path}/{filename}.json'))
    return config


def is_available_TPU():
    """
    Is TPU available?
    """
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        TPU = True
        os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
        os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'
    except:
        TPU = False

    return TPU


def getDevice():
    """
    Get compute device: TPU/GPU/CPU
    """
    if is_available_TPU():
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif torch.cuda.is_available():
         device = 'cuda'
    else:
         device = 'cpu'
    
    print(f"Using {device} device.")
    return device


def setSeeds(seed_val=0, cudnn_active=False):
    """
    Set relevant seeds for reproducibility (as much as possible, of course)

    seed_val: an integer (default 0)
    cudnn_active: whether to do cudnn benchmarking to be active (default to False);
                  Keeping it False, however, might slowdown a bit; but guarantees better reproducibility
    """
    random.seed(seed_val)
    rs = RandomState(MT19937(SeedSequence(seed_val)))
    np.random.seed(seed_val)             
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val) #for single GPU
    torch.cuda.manual_seed_all(seed_val) #for multi-GPU 

    if not cudnn_active:
        torch.backends.cudnn.benchmark = False    
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.enabled = False

    print(f'Random seed set to {seed_val}; Cudnn benchmarking is {cudnn_active}.')
