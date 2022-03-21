import torch
import torch.nn as nn

out_shape = 4
class LSTMPointPredict(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, drop_prob, device, is_qat, offset, **kwargs):

        super(LSTMPointPredict, self).__init__()

        self.is_qat = is_qat
        self.device = device
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.encoder = nn.LSTM(input_size=3,
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
        self.full_connected = nn.Linear(dec_hidden_size, out_shape)
        self.crit = nn.MSELoss()

        if is_qat:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        self.v_matrix = torch.tensor([[1, 0],
                                     [0, 0],
                                     [0, 1],
                                     [0, 0]], requires_grad=False).to(device)

        self.a_matrix = torch.tensor([[0, 0],
                                     [1, 0],
                                     [0, 0],
                                     [0, 1]], requires_grad=False).to(device)

    def forward(self,
                inp):

        # input data shape should be (batch_size, 10, 3), same as their paper
        inp_tensor = inp['input'].to(self.device)
        if self.is_qat:
            inp_tensor = self.quant(inp_tensor)

        out, hc = self.encoder(inp_tensor)
        out = self.drop(out)
        out, _ = self.decoder(out) # add hc ?
        out = self.drop(out)
        logits = self.full_connected(out[:, -1]) # v_x, a_x, v_y, a_y
        if self.is_qat:
            logits = self.dequant(logits)
        if out_shape == 4:
            dt = inp['dt'][:, None, None].to(self.device)
            dis_matrix = torch.mul(dt, self.v_matrix) + 0.5 * torch.pow(torch.mul(dt, self.a_matrix), 2)
            pred_x_y = torch.bmm(logits.unsqueeze(1), dis_matrix)
            loss = self.crit(pred_x_y.squeeze(1), inp['label'].to(self.device))
        else:
            loss = self.crit(logits, inp['label'].to(self.device))

        return logits, loss
