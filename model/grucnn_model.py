import torch
import torch.nn as nn

class GRUCNNPointPredict(nn.Module):
    def __init__(self, enc_hs, dec_hs, drop_prob, device, **kwargs):
        super(GRUCNNPointPredict, self).__init__()
        self.device = device

        self.gru = nn.GRU(input_size=3,
                          hidden_size=64,
                          batch_first=True)
        self.bn = nn.BatchNorm1d(40)

        self.cnn = nn.Sequential(nn.Conv1d(in_channels=40,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding='same'),
                                 nn.PReLU(),
                                 nn.Conv1d(in_channels=32,
                                           out_channels=64,
                                           kernel_size=3,
                                           padding='same'),
                                 nn.PReLU(),
                                 nn.Conv1d(in_channels=64,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding='same'),
                                 nn.PReLU())
        self.cnn_2 = nn.Conv1d(in_channels=1,
                               out_channels=64,
                               kernel_size=1)
        self.gap = nn.AdaptiveAvgPool1d(1)


        self.ffc = nn.Linear(160, 2)
        self.crit = nn.MSELoss()

    def forward(self, inp):

        inp_tensor = inp['input'].to(self.device)
        out_1, _ = self.gru(inp_tensor.detach().clone())
        out_1 = self.bn(out_1)
        out_2 = self.cnn(inp_tensor.detach().clone())

        out_2_1 = self.gap(out_2)
        out_2_2 = torch.max(out_2, dim=1, keepdim=True)[0]
        out_2_2 = self.cnn_2(out_2_2)
        out_2_2 = self.gap(out_2_2)
        out = torch.cat((out_1[:, -1], out_2_1.squeeze(-1), out_2_2.squeeze(-1)), dim=1)
        logits = self.ffc(out)
        loss = self.crit(logits, inp['label'].to(self.device))

        return logits, loss







