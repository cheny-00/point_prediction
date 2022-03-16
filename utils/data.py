
import os
import math
import json

import torch
from tqdm import tqdm
import pandas as pd
import pickle
from torch.utils.data.dataset import Dataset


n_prev = 40

class LSTMSample:
    def __init__(self, events, sequence_counter, count, jump):
        self.angle = 0

        self.x = []
        self.y = []
        self.time = []
        for i in range(sequence_counter, sequence_counter + count):  # padding
            if i >= jump - 1:
                self.x.append(events[jump - 1][0])
                self.y.append(events[jump - 1][1])

                if i == sequence_counter:
                    self.time.append(4)
                else:
                    self.time.append(events[jump - 1][3] - events[jump - 2][3])
            else:
                self.x.append(events[i][0])
                self.y.append(events[i][1])
                if i == sequence_counter:
                    self.time.append(4)
                else:
                    self.time.append(events[i][3] - events[i - 1][3])

    def derivate(self):
        x_der = []
        y_der = []
        time_der = []
        for i in range(len(self.x) - 1):
            x_der.append(self.x[i + 1] - self.x[i])
            y_der.append(self.y[i + 1] - self.y[i])
            time_der.append(self.time[i + 1])
        self.x = x_der
        self.y = y_der
        self.time = time_der

    def getAngle(self, startIndex, endIndex):
        myX = self.x[endIndex] - self.x[startIndex]
        myY = self.y[endIndex] - self.y[startIndex]
        return math.atan2(myX, myY)

    def rotate(self, angle, originIndex):
        self.rotation = angle

        cs = math.cos(angle)
        sn = math.sin(angle)
        for i in range(len(self.x)):
            myX = (self.x[i] - self.x[originIndex]) * cs - \
                  (self.y[i] - self.y[originIndex]) * sn
            myY = (self.x[i] - self.x[originIndex]) * sn + \
                  (self.y[i] - self.y[originIndex]) * cs
            self.x[i] = myX + self.x[originIndex]
            self.y[i] = myY + self.y[originIndex]


class LSTMDataset(Dataset):

    def __init__(self,
                 data_path,
                 offset):

        self.data_path = data_path
        self.data = list()
        self.target = list()
        self.dt = list()
        self.offset = offset


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.get_item(self.data[idx], self.target[idx], self.dt[idx])

    def get_data(self):
        events = self.read_data(self.data_path)
        raw_sample = self.load_dataset(events, n_prev + 1 + self.offset)
        self.data, self.target, self.dt = self.build_samples(raw_sample, self.offset)

    @staticmethod
    def get_item(data, target, dt):
        features = dict()
        features['input'] = torch.tensor(data, dtype=torch.float32)
        features['label'] = torch.tensor(target, dtype=torch.float32)
        features['dt'] = torch.tensor(dt, dtype=torch.float32)

        return features


    @staticmethod
    def build_samples(raw_samples, offset):
        data, target, dt = list(), list(), list()

        for sample in raw_samples:
            line = list()
            for i in range(n_prev):
                line.append([sample.x[i], sample.y[i], sample.time[i + offset]])
            data.append(line)
            x = y = t = 0
            for i in range(offset):
                x += sample.x[n_prev + i]
                y += sample.y[n_prev + i]
                t += sample.t[n_prev + i]
            target.append([x, y])
            dt.append(t)
        return data, target, dt

    @staticmethod
    def load_dataset(raw_data, seq_len):
        # return shape should be (N, seq_len, 3)
        samples = list()
        for i, event in enumerate(raw_data):
            jump = i + seq_len + 1
            for j in range(i, i + seq_len):
                if j >= len(raw_data) or raw_data[j][4] != 2:
                    jump = j
                    break
            if jump - i > n_prev + 1:
                sample = LSTMSample(raw_data, i, seq_len, jump)
                sample.angle = sample.getAngle(n_prev - 1, n_prev) + math.radians(45)
                sample.rotate(sample.angle, n_prev)

                sample.derivate()

                same_time = 0
                for a in range(len(sample.time)):
                    if sample.time[a] < 1:
                        same_time = 1
                if same_time == 0:
                    samples.append(sample)
        return samples

    @staticmethod
    def read_data(data_path):
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
        for i in tqdm(range(0, len(split_points) - 1, 2)):
            start, end = split_points[i] + 1, split_points[i + 1]
            if i == 50:
                break
            for idx, p in raw_data.iloc[start:end].iterrows():
                action = 0 if idx == start else 1 if idx == end - 1 else 2
                events.append([p.x, p.y, p.total_tilt, p.timestamp, action])
        return events


def load_json_data(data_path):
    data = json.load(open(data_path, "r"))
    events = list()
    for s in data:
        points = s['mPoints']
        start, end = 0, len(points)
        for pos, point in enumerate(points):
            if not "time" in point.keys():
                print("Not found timestamp attribute")
            action = 0 if pos == start else 1 if pos == end else 2
            events.append([point['x'], point['y'], 0, point['time'],action])

    return events

def save_dataset(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



