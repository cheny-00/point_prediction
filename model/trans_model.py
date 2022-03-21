import torch
import torch.nn as nn

class TransPointPredict(nn.Module):
    def __init__(self, enc_hs, dec_hs, drop_prob, device, **kwargs):
        super(TransPointPredict, self).__init__()

        self.device = device
        self.ffn_1 = nn.Linear(3, enc_hs)
        self.drop = nn.Dropout(drop_prob)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=enc_hs,
                                                        nhead=8,
                                                        activation='gelu',
                                                        batch_first=True,
                                                        norm_first=True)
        self.batch_norm_1 = nn.BatchNorm1d(40)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=6,
                                             norm=self.batch_norm_1)
        self.ffn_2 = nn.Sequential(nn.Linear(enc_hs, dec_hs),
                                   nn.PReLU(),
                                   self.drop)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=dec_hs,
                                                        nhead=8,
                                                        activation='gelu',
                                                        batch_first=True,
                                                        norm_first=True)

        self.batch_norm_2 = nn.BatchNorm1d(40)
        self.decoder = nn.TransformerEncoder(encoder_layer=self.decoder_layer,
                                             num_layers=6,
                                             norm=self.batch_norm_2)

        self.ffc = nn.Sequential(nn.Linear(dec_hs, 4),
                                 nn.Tanh(),
                                 self.drop,
                                 nn.Linear(4, 2))

        self.crit = nn.MSELoss()


    def forward(self, inp):

        inp_tensor = inp['input'].to(self.device)
        emb = self.ffn_1(inp_tensor)
        out = self.encoder(emb)
        out = self.drop(out)
        out = self.ffn_2(out)
        out = self.decoder(out)
        logits = self.ffc(out[:, -1])
        loss = self.crit(logits, inp['label'].to(self.device))

        return logits, loss
