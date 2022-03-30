from model.lstm_model import LSTMPointPredict
from model.trans_model import TransPointPredict
from model.grucnn_model import GRUCNNPointPredict
from model.lstm_depressed_model import LSTMDepressedPointPredict
models = {
    "lstm": LSTMPointPredict,
    "trans": TransPointPredict,
    "grucnn": GRUCNNPointPredict,
    "lstm_depressed": LSTMDepressedPointPredict
}