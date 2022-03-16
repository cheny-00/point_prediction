import torch
import torch.nn as nn

class LSTMPointPredict(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, drop_prob, device, is_qat):

        super(LSTMPointPredict, self).__init__()

        self.is_qat = is_qat
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
        self.full_connected = nn.Linear(dec_hidden_size, 4)
        self.crit = nn.MSELoss()
        if is_qat:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self,
                inp,
                dt):

        # input data shape should be (batch_size, 10, 3), same as their paper
        inp_tensor = inp['input'].to(self.device)
        if self.is_qat:
            inp_tensor = self.quant(inp_tensor)
        dt_p2 = 0.5 * dt * dt

        out, hc = self.encoder(inp_tensor)
        out = self.drop(out)
        out, _ = self.decoder(out) # add hc ?
        out = self.drop(out)
        logits = self.full_connected(out[:, -1]) # v_x, a_x, v_y, a_y
        if self.is_qat:
            logits = self.dequant(logits)
        cal_coordinate_matrix = torch.tensor([[dt, 0],
                                              [dt_p2 , 0],
                                              [0, dt],
                                              [0, dt_p2]])

        pred_x_y = torch.mm(logits, cal_coordinate_matrix)

        loss = self.crit(pred_x_y, inp['label'].to(self.device))

        return pred_x_y, loss
