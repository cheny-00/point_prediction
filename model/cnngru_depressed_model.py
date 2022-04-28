import torch
import torch.nn as nn
from .depressed_model import DepressedPredictor

class CNNGRUDepressedPointPredict(DepressedPredictor):
    def __init__(self, **kwargs):

        super(CNNGRUDepressedPointPredict, self).__init__()

        if 'is_qat' in kwargs:
            self.is_qat = kwargs['is_qat']
        else:
            self.is_qat = False
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.enc_hidden_size = kwargs['enc_hidden_size']
        self.dec_hidden_size = kwargs['dec_hidden_size']

        self.offset = kwargs['offset']
        self.scales = torch.linspace(1 / self.offset, 1, self.offset, dtype=torch.float32, device=self.device)

        dec_hidden_size = 32

        self.conv_layers = nn.Sequential(nn.Conv1d(25, 32, 7, padding=3),
                                         nn.ReLU(),
                                         nn.Conv1d(32, 64, 3, padding=1),
                                         nn.ReLU(),
                                         nn.GRU(input_size=5,
                                                hidden_size=32,
                                                batch_first=True))

        # self.drop = nn.Dropout(drop_prob)

        self.va_ffc = nn.Linear(dec_hidden_size, 4)
        self.time_prediction = nn.Sequential(nn.Linear(dec_hidden_size, 1),
                                             nn.Sigmoid())


    def forward(self,
                inp):

        # input data shape should be (batch_size, 10, 3), same as their paper
        inp_tensor = inp['input'].to(self.device)
        time_to_predict = torch.tile(self.scales, (len(inp_tensor), 1))

        out, _ = self.conv_layers(inp_tensor)

        polynomial = self.va_ffc(out[:, -1]).reshape(-1, 2, 2)
        predicted_time = self.time_prediction(out[:, -1])

        powers = torch.cat((predicted_time, torch.square(predicted_time)), -1)
        prediction = torch.matmul(powers[:, None, :], polynomial).squeeze(1)
        powers = torch.stack((time_to_predict, torch.square(time_to_predict)), -1)
        projection = torch.matmul(powers[:, :, None, :], polynomial[:, None, :, :]).squeeze(-2)

        loss, loss_details = self.loss_fuc(prediction, projection, inp['targets'].to(self.device), predicted_time, time_to_predict)
        return prediction, loss, loss_details
