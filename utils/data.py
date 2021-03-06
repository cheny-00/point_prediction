
import os
import math
import json

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import pickle
from copy import deepcopy
from torch.utils.data.dataset import Dataset


n_prev = 25
TWO_PI = 2 * math.pi

class LSTMDataset(Dataset):

    def __init__(self,
                 data_path,
                 offset,
                 use_n_data,
                 eval_step=False,
                 save_data_path=None):

        self.targets = None
        self.cum_time = list()
        self.data_path = data_path
        self.data = list()
        self.target = list()
        self.dt = list()
        self.offset = offset
        self.normalize_event_count = 5
        self.do_cum_time = True
        self.do_sum_targets = True
        self.default_time_indent = 1000 / 240
        self.time_diff_tolerance = 3.5
        self.eval_step = eval_step
        self.use_n_data = use_n_data
        if save_data_path is None:
            save_data_path = "/home/cy/workspace/npp/data/raw_dataset.npy"
        self.save_data_path = save_data_path
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.get_item(input=self.data[idx],
                             label=self.target[idx],
                             dt=self.dt[idx],
                             cum_time=self.cum_time[idx],
                             targets=self.targets[idx])

    def get_data(self):
        # events shape will be (N, seq_len, features)
        # 1. split event
        # 2. normalize
        if not os.path.exists(self.save_data_path):
            events = self.read_data(self.data_path, use_n_data=self.use_n_data, offset=self.offset)
            raw_dataset = self.build_dataset(events)
            with open(self.save_data_path, 'wb') as np_f:
                np.save(np_f, raw_dataset)
        else:
            raw_dataset = np.load(self.save_data_path)
        
        self.data, self.target, self.dt, self.cum_time, self.targets = self.build_samples(raw_dataset, self.offset)

    def build_dataset(self, events):
        raw_dataset = list()
        total_seq_len = n_prev + self.offset + 2
        for event in tqdm(events):
            n = event.shape[0]
            
            while total_seq_len < n:
                n_data = event[n - total_seq_len:n]
                # n_data = self.smooth_diff(n_data)
                n_data = self.linear_interpolation(n_data)
                n_data = self.normalize(n_data)
                n -= 1
                if n_data is not None:
                    raw_dataset.append(n_data)
                if self.eval_step: # just collect a event once when we are doing evaluation.
                    break
        print("total dataset size: ", len(raw_dataset))
        return raw_dataset

    @staticmethod
    def get_item(**kwargs):
        features = dict()
        for k, v in kwargs.items():
            if v is not None:
                features[k] = torch.tensor(v, dtype=torch.float32)

        return features

    def build_samples(self, raw_dataset, offset):

        raw_dataset = np.array(raw_dataset) # shape will be (N, seq_len, features)
        assert raw_dataset.shape[1] >= offset + n_prev, f"{raw_dataset.shape[1]} < {offset + n_prev}"
        data = raw_dataset[:, :n_prev ,:]
        label = np.sum(deepcopy(raw_dataset[:, n_prev:, :2]), axis=1)
        targets = np.cumsum(deepcopy(raw_dataset[:, n_prev:, :2]), axis=1)
        label_dt = np.sum(deepcopy(raw_dataset[:, n_prev:, 2]), axis=1)
        label_cum_time = np.cumsum(deepcopy(raw_dataset[:, n_prev:, 2]), axis=1)

        return data, label, label_dt, label_cum_time, targets

    def normalize(self, trace, add_noise=False):
        if trace is None:
            return trace
        normal_direction = trace[n_prev - 1,:2] - trace[n_prev - 1 - self.normalize_event_count, :2]
        diff_trace = trace[1:] - trace[:-1]
        # print("before", diff_trace)
        # print("before", diff_trace)
        dxy = diff_trace[:, :2]
        dt = diff_trace[:, 2]
        dp = diff_trace[:, 3]
        do = diff_trace[:, 4:]

        if add_noise:
            dxy *= (1 + 0.10 * np.random.normal())
        angle = math.atan2(normal_direction[0], normal_direction[1]) + math.radians(45)
        cs = math.cos(angle)
        sn = math.sin(angle)
        dxy = np.dot(dxy, np.array([[cs, sn], [-sn, cs]]))
        two_pi = 2 * np.pi
        do = do - np.round(do / two_pi) * two_pi
        # feature = np.hstack([dxy, dt[:, None], dp[:, None], do])

        feature = np.hstack([dxy, dp[:, None], do])
        return feature
    
    def linear_interpolation(self, feature):
        if feature is None:
            return None
        n, i = feature.shape[0], 0
        t_0 = feature[0, 2]
        norm_feature = list()
        while i < n - 1:
            last_t = feature[i, 2] - t_0
            next_t = feature[i + 1, 2] - t_0
            if next_t - last_t == 0:
                return None
            alpha = (next_t - self.default_time_indent * (i + 1)) / (next_t - last_t)
            beta = 1 - alpha
            norm_feature.append(feature[i] * alpha + feature[i + 1] * beta)
            i += 1
        norm_feature = np.array(norm_feature)
        norm_feature[:, 2] = norm_feature[:, 2] - norm_feature[0, 2]
        # print(np.array(norm_feature).shape)
        return norm_feature

    
    def smooth_diff(self, feature):
        dt = feature[1:] - feature[:-1]
        max_dt, min_dt = max(dt[:, 2]), min(dt[:, 2])
        if max_dt > self.default_time_indent + self.time_diff_tolerance or min_dt < self.default_time_indent -  self.time_diff_tolerance:
            return None
        return feature



    @staticmethod
    def read_data(data_path, use_n_data, offset=8):
        # return [x, y, angle(no need), time, action]
        header_name =  ["x", "y", "pressure", "x_tilt", "y_tilt", "total_tilt", "orientation", "pitch_angle", "scroll",
                        "pen_size", "pen_size_x", "pen_size_y", "pen_size_width", "pen_size_height" ,"timestamp"]
        raw_data = pd.read_csv(data_path,
                               sep=" ",
                               header=len(header_name),
                               names=header_name)
        split_points = [-1] + raw_data[raw_data.y.isna()].x.index.tolist()
        split_points = [int(n) for n in split_points]
        events = list()
        n_event = 0
        for i in tqdm(range(0, len(split_points) - 1, 2)):
            start, end = split_points[i] + 1, split_points[i + 1]
            n_event += 1
            if len(events) == use_n_data:
                break
            event = list()
            for idx, p in raw_data.iloc[start:end].iterrows():
                event.append([p.x, p.y, p.timestamp, p.pressure, p.total_tilt, p.orientation])
            if len(event) > offset + n_prev + 2:
                events.append(np.array(event))
        print(f"{len(events)} / {n_event}")
        return events



def save_dataset(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


