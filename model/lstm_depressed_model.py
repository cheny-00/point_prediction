import torch
import torch.nn as nn
from depressed_model import DepressedPredictor

class LSTMDepressedPointPredict(DepressedPredictor):
    def __init__(self, enc_hidden_size, dec_hidden_size, drop_prob, **kwargs):

        super(LSTMDepressedPointPredict, self).__init__()

        self.encoder = nn.LSTM(input_size=5,
                               hidden_size=enc_hidden_size,
                               num_layers=1,
                               batch_first=True,
                               bias=True)

        self.drop = nn.Dropout(drop_prob)

        self.decoder = nn.LSTM(input_size=enc_hidden_size,
                               hidden_size=dec_hidden_size,
                               num_layers=1,
                               batch_first=True,
                               bias=True)

        self.va_ffc = nn.Linear(dec_hidden_size, 4)
        self.time_prediction = nn.Sequential(nn.Linear(dec_hidden_size, 1),
                                             nn.Sigmoid())

    def forward(self,
                inp):

        # input data shape should be (batch_size, 10, 3), same as their paper
        inp_tensor = inp['input'].to(self.device)
        time_to_predict = torch.tile(self.scales, (len(inp_tensor), 1))

        out, hc = self.encoder(inp_tensor)
        out = self.drop(out)
        out, _ = self.decoder(out) # add hc ?
        out = self.drop(out)

        polynomial = self.va_ffc(out[:, -1]).reshape(-1, 2, 2)
        predicted_time = self.time_prediction(out[:, -1])


        powers = torch.cat((predicted_time, torch.square(predicted_time)), -1)
        prediction = torch.matmul(powers[:, None, :], polynomial).squeeze(1)
        powers = torch.stack((time_to_predict, torch.square(time_to_predict)), -1)
        projection = torch.matmul(powers[:, :, None, :], polynomial[:, None, :, :]).squeeze(-2)

        loss = self.loss_fuc(prediction, projection, inp['targets'].to(self.device), predicted_time, time_to_predict)
        return prediction, loss, predicted_time
