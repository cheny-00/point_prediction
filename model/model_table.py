from model.lstm_model import LSTMPointPredict
from model.trans_model import TransPointPredict
models = {
    "lstm": LSTMPointPredict,
    "trans": TransPointPredict
}