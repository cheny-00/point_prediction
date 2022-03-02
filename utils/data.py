
import os

import torch
import pickle

from tqdm import tqdm
from functools import partial, reduce
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def load_data():
    # wait for data format
    pass


class SamsungDataset(Dataset):
