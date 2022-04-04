from model.lstm_model import LSTMPointPredict
from model.trans_model import TransPointPredict
from model.grucnn_model import GRUCNNPointPredict
from model.lstm_depressed_model import LSTMDepressedPointPredict
from model.cnngru_depressed_model import CNNGRUDepressedPointPredict
models = {
    "lstm": LSTMPointPredict,
    "trans": TransPointPredict,
    "grucnn": GRUCNNPointPredict,
    "lstm_depressed": LSTMDepressedPointPredict,
    "cnngru_depressed": CNNGRUDepressedPointPredict
}