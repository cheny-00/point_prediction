import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from load_args import load_args
from utils.data import LSTMDataset
from utils.utils import create_exp_dir
from model.model import LSTMPointPredict
from utils.trainer import LSTMModelTrainer


#############################################################################################
##  setting
#############################################################################################
args = load_args()

start_time = time.strftime('%Y%m%d-%H%M%S')
work_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'logs')
log_dir = os.path.join(work_dir, args.proj_name, start_time)
working_files  = ["run_train.py", "load_args.py", "model/model.py", "utils/trainer.py"]
working_files = list(map(lambda x: os.path.join(work_dir, x), working_files))
logging = create_exp_dir(log_dir,
                         scripts_to_save=working_files,
                         debug=args.debug)

device = torch.device('cuda' if args.cuda else 'cpu')
if torch.cuda.is_available() and not args.cuda:
    print("detected gpu is available but not using")

#############################################################################################
##  loading dataset
#############################################################################################


train_data = LSTMDataset(data_path=args.data_path,
                         offset=args.offset)
train_dataloader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              drop_last=False,
                              shuffle=True)

model = LSTMPointPredict(enc_hidden_size=512,
                         dec_hidden_size=256,
                         drop_prob=args.dropout,
                         device=device)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=args.lr)

#############################################################################################
##  train model
#############################################################################################

trainer_params = {
    "epochs": args.epochs,
    "train_iter": train_dataloader,
    "dev_iter": None,
    "logger": logging,
    "batch_size": args.batch_size,
    "log_interval": args.log_interval,
    "eval_interval": args.eval_interval,
    "optim": optim
}

model_trainer = LSTMModelTrainer(**trainer_params)

model_trainer.train(model, None)

#############################################################################################
##  save model
#############################################################################################
save_params = {
    "state_dict": model.state_dict(),
    "optimizer": optim.state_dict(),
}
save_path = os.path.join(work_dir, "save", "LSTM", start_time)

torch.save(save_params,
           save_path)




