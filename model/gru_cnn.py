import torch
import torch.nn as nn

class CNNGRUPointPredict(nn.Module):
    def __init__(self, enc_hs, dec_hs, drop_prob, device):
        self.device = device

        self.gru = nn.GRU(input_size=3,
                          hidden_size=64,
                          batch_first=True)

        self.batchnorm_1 = nn.BatchNorm1d(40)

        self.cnn = nn.Sequential(nn.Conv1d(in_channels=3,
                                           out_channels=32,
                                           kernel_size=15),
                                 nn.PReLU(),
                                 nn.Conv1d(in_channels=32,
                                           out_channels=64,
                                           kernel_size=7),
                                 nn.PReLU(),
                                 nn.Conv1d(in_channels=64,
                                           out_channels=32,
                                           kernel_size=3),
                                 nn.PReLU())


        self.gap = lambda x: nn.functional.avg_pool2d(x, (1, 1))




