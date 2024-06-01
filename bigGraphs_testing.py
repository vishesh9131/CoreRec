import numpy as np
import vish_graphs as vg 
import core_rec as cr 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    num_people=10000
    # num_people=100
    file_path = vg.generate_large_random_graph(num_people, file_path="blobbag.csv",seed=56)
