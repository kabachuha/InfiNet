import torch
if not torch.__version__.startswith('2'):
    print("Please, use torch 2 when training, or else you'll lose a lot of compute :)")
    exit()
