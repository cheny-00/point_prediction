import torch
import torch.nn as nn

class LSTMPointPredict(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, drop_prob, device):

        super(LSTMPointPredict).__init__()

        self.device = device
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.encoder = nn.LSTM(input_size=3,
                               hidden_size=enc_hidden_size,
                               num_layers=1,
                               bias=True)

        self.drop = nn.Dropout(drop_prob)

        self.decoder = nn.LSTM(input_size=enc_hidden_size,
                               hidden_size=dec_hidden_size,
                               num_layers=1,
                               bias=True)
        self.full_connected = nn.Linear(dec_hidden_size, 2)
        self.crit = nn.MSELoss()

    def forward(self,
                inp):

        # input data shape should be (batch_size, 10, 3), same as their paper

        out, hc = self.encoder(inp['input'].to(self.device))
        out = self.drop(out)
        out, _ = self.decoder(out) # add hc ?
        out = self.drop(out)
        logits = self.full_connected(out)
        loss = self.crit(logits, inp['label'].to(self.device))

        return logits, loss

