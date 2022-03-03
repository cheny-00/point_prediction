import os
import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from load_args import load_args
from utils.data import LSTMDataset
from utils.utils import create_exp_dir, randn_sampler
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
##  load dataset
#############################################################################################
val_split = [0.1, 0.2]


dataset = LSTMDataset(data_path=args.data_path,
                         offset=args.offset)
dev_sampler, eval_sampler, train_sampler = randn_sampler(val_split,
                                                         len(dataset),
                                                         shuffle_dataset=True,
                                                         random_seed=args.seed)



train_iter = DataLoader(dataset,
                        batch_size=args.batch_size,
                        drop_last=False,
                        shuffle=True,
                        sampler=train_sampler)
dev_iter = DataLoader(dataset,
                      batch_size=args.eval_batch_size,
                      sampler=dev_sampler)
eval_iter = DataLoader(dataset,
                       batch_size=args.eval_batch_size,
                       sampler=eval_sampler)


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
    "train_iter": train_iter,
    "dev_iter": dev_iter,
    "logger": logging,
    "batch_size": args.batch_size,
    "log_interval": args.log_interval,
    "eval_interval": args.eval_interval,
    "optim": optim
}

model_trainer = LSTMModelTrainer(**trainer_params)

model_trainer.train(model, None)

#############################################################################################
##  evaluate model
#############################################################################################

model.eval()
tqdm_eval_iter = tqdm(eval_iter)
total_eval_score = 0
eval_start_time = time.time()
with torch.no_grad():
    for data in enumerate(tqdm_eval_iter):
        _, eval_score = model(data)
        total_eval_score += eval_score

eval_avg_score = total_eval_score / len(tqdm_eval_iter)
eval_log_str  = f"| Final Eval | speed time: {time.time() - eval_start_time} |" \
                f"| average eval loss: {eval_avg_score} | "

logging('-' * 100 + "\n" + eval_log_str + "\n" + '-' * 100, print_=True)


#############################################################################################
##  save model
#############################################################################################
save_params = {
    "state_dict": model.state_dict(),
    "optimizer": optim.state_dict(),
    "average_eval_score": eval_avg_score
}
save_path = os.path.join(work_dir, "save", "LSTM", start_time)

torch.save(save_params,
           save_path)





