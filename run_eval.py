import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.eval_tools import ade
from utils.data import LSTMDataset
from model.lstm_model import LSTMPointPredict
from model.lstm_depressed_model_eval import LSTMDepressedPointPredict

gpu_rank = 5
offset_list = [8]
batch_size = 1
# eval_data_path = "/home/cy/workspace/datasets/0311_data_4ms/draw_5_copies"
eval_data_path = "/home/cy/workspace/datasets/new_data_shapes/draw_5_copies"
#eval_data_path = "/home/cy/workspace/datasets/0311_new_data/draw_5_copies"
#ckpt_path = "/home/cy/workspace/npp/save/LSTM/20220312-100635/epoch_272_74.466.pt"
#ckpt_path = "/home/cy/workspace/npp/save/LSTM/20220311-093403/epoch_250_39.179.pt"
#ckpt_path = "/home/cy/workspace/npp/save/LSTM/20220312-155456/epoch_968_39.951.pt"
#ckpt_path = "/home/cy/workspace/npp/save/LSTM/20220312-155439/LSTM.pt"
# ckpt_path = "/home/cy/workspace/npp/save/LSTM/20220313-022038/LSTM.pt"
# ckpt_path = "/home/cy/workspace/npp/save/lstm_depressed/20220323/170858/epoch_200_0.028.pt"
ckpt_path = "/home/cy/workspace/npp/save/lstm_depressed/20220324/182439/epoch_160_0.028.pt"
device = torch.device(f'cuda:{gpu_rank}' if torch.cuda.is_available() else 'cpu')

# model = LSTMPointPredict(enc_hidden_size=512,
#                          dec_hidden_size=256,
#                          drop_prob=0.75,
#                          device=device)
model = LSTMDepressedPointPredict(128, dec_hidden_size=64, drop_prob=0.75, device=device, is_qat=False)
ckpt_path = torch.load(ckpt_path)
model.load_state_dict(ckpt_path['model_state_dict'])

model.to(device)
model.eval()

for offset in offset_list:
    print(f"eval dataset from: {eval_data_path}")
    eval_dataset = LSTMDataset(data_path=eval_data_path,
                                offset=offset)
    eval_dataset.get_data()
    eval_iter = DataLoader(eval_dataset,
                           batch_size=batch_size)

    tqdm_eval_iter = tqdm(eval_iter)
    total_dist = total_predicted_time = 0
    for data in tqdm_eval_iter:
        target = data['label'].to(device)
        pred_time, logits = model(data)
        total_predicted_time += pred_time.item
        total_dist += ade(logits, target)
    print(f"{offset * 4} ms {total_dist / len(eval_iter)} px")
    print(f"{offset * 4} ms {total_predicted_time / len(eval_iter)} ms")
