import os
import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from load_args import load_args
from utils.data import LSTMDataset
from utils.utils import create_exp_dir, randn_sampler
from utils.trainer import LSTMModelTrainer, DepressModelTrainer
from model.model_table import *



import pickle

data = pickle.load(open("ck_data.pkl", 'rb'))
train_set = data['train_set'][0][0][0][0]
train_data = list()
train_label = list()
batch_size = 256
train_iter = list()
for i, sample in enumerate(train_set):
    sample_data, sample_label, sample_time = sample.get_feature_caption()
    train_data.append(sample_data)
    train_label.append(sample_label)
    if i + 1 == batch_size:
        feature = dict()
        feature['input'] = torch.tensor(train_data, dtype=torch.float)
        feature['targets'] = torch.tensor(train_label, dtype=torch.float)
        train_iter.append(feature)
        train_data = list()
        train_label = list()
valid_set = data['valid_set']
valid_data = list()
valid_label = list()
valid_iter = list()
for i, sample in enumerate(valid_set):
    valid_data.append(sample.feature)
    valid_label.append(sample.caption)
    if i + 1 == batch_size:
        feature = dict()
        feature['input'] = torch.tensor(valid_data, dtype=torch.float)
        feature['targets'] = torch.tensor(valid_label, dtype=torch.float)
        valid_iter.append(feature)
        valid_data = list()
        valid_label = list()

pickle.dump(train_iter, open('ck_train_data.pkl', 'wb'))
pickle.dump(valid_iter, open('ck_val_data.pkl', 'wb'))




